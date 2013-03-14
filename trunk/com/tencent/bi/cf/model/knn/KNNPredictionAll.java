package com.tencent.bi.cf.model.knn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.LongPairSecondaryComparator;
import com.tencent.bi.utils.serialization.KeyPairWritable;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.TripletWritable;

/**
 * Class for predicting all user/item pairs
 * @author tigerzhong
 *
 */
public class KNNPredictionAll {
	/**
	 * Path for training data
	 * @param trainingPath, path
	 * @param userBase, if user-base or item-base
	 * @throws Exception
	 */
	public static void predict(String trainingPath, boolean userBase) throws Exception{
		Configuration conf = FileOperators.getConfiguration(); 
		conf.setBoolean("model.userbase", userBase);
		conf.set("mapred.textoutputformat.separator", ",");
		//Combine
		Job job = new Job(conf);
		job.setJobName("KNN-Prediction-Join");
		job.setJarByClass(KNNPredictionAll.class);
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(KeyPairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(TripletWritable.class);
		job.setMapperClass(CombineMapper.class);
		job.setReducerClass(CombineReducer.class);
		job.setPartitionerClass(FirstPartitioner.class);
		job.setSortComparatorClass(LongPairSecondaryComparator.KeyComparator.class);
		job.setGroupingComparatorClass(LongPairSecondaryComparator.GroupComparator.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(trainingPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"NgList/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNCombine/"));
		job.waitForCompletion(true);
		//Reverse
		job = new Job(conf);
		job.setJobName("KNN-Prediction-Compute");
		job.setJarByClass(NgBuilder.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(TripletWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(ComputeReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNCombine/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"KNN-Prediction/"));
		job.waitForCompletion(true);
	}
	
	/**
	 * Partition class for assigning data to reduce machines 
	 * @author tigerzhong
	 *
	 */
	public static class FirstPartitioner extends Partitioner<LongPairWritable, KeyPairWritable> {

		HashPartitioner<Long, KeyPairWritable> p = new HashPartitioner<Long, KeyPairWritable>();
		
		@Override
		public int getPartition(LongPairWritable key, KeyPairWritable value,
				int numPartitions) {
			return p.getPartition(key.getFirst(), value, numPartitions);
		}

	}
	
	/**
	 * Mapper for combining neighbor information and existed data
	 * @author tigerzhong
	 *
	 */
	public static class CombineMapper extends Mapper<LongWritable, Text, LongPairWritable, KeyPairWritable> {
		/**
		 * Key for output
		 */
		private LongPairWritable outKey = new LongPairWritable();
		
		/**
		 * Value for output
		 */
		private KeyPairWritable outValue = new KeyPairWritable();

		private boolean userBase = true;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			if(path.indexOf("/NgList")>-1){
				if(userBase) {
					outKey.set(Long.parseLong(items[0]),1);
					outValue.set(Long.parseLong(items[1]), Double.parseDouble(items[2]));
				} else {
					outKey.set(Long.parseLong(items[1]),1);
					outValue.set(Long.parseLong(items[0]), Double.parseDouble(items[2]));
				}
			} else {
				for(int i=1;i<items.length;i+=2){
					outKey.set(Long.parseLong(items[i]),0);
					outValue.set(Long.parseLong(items[0]), Double.parseDouble(items[i+1]));
				}
			}			
			context.write(outKey, outValue);
		}
		
		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			userBase = conf.getBoolean("model.userbase", true);
		}
	}
	
	/**
	 * Reducer for combining neighbor information and existed data
	 * @author tigerzhong
	 *
	 */
	public static class CombineReducer extends Reducer<LongPairWritable, KeyPairWritable, LongWritable, TripletWritable> {
		
		/**
		 * Value for output
		 */
		private TripletWritable outValue = new TripletWritable();
		
		private LongWritable outKey = new LongWritable();
		
		protected List<Long> idList = new ArrayList<Long>();
		
		protected List<Double> valList = new ArrayList<Double>();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<KeyPairWritable> values, Context context)
				throws IOException, InterruptedException {
			for(KeyPairWritable value : values){
				if(key.getSecond()==0){//NgList
					idList.add(value.getKey());
					valList.add(value.getValue());
				} else {
					long id = value.getKey();
					double r = value.getValue();
					for(int i=0;i<idList.size();i++){
						outKey.set(idList.get(i));
						outValue.set(id, valList.get(i), r);
						context.write(outKey, outValue);
					}
				}
			}
		}
	}

	/**
	 * Reducer for computing the prediction results
	 * @author tigerzhong
	 *
	 */
	public static class ComputeReducer extends Reducer<LongWritable, TripletWritable, LongWritable, Text> {
		
		/**
		 * Value for output
		 */
		private Text outValue = new Text();
		
		private HashMap<Long, Double> sumMap = null;
		
		private HashMap<Long, Double> simMap = null;
		
		private int n = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<TripletWritable> values, Context context)
				throws IOException, InterruptedException {
			for(TripletWritable value : values){
				long k = value.getP();
				if (sumMap.containsKey(k)){
					sumMap.put(k, sumMap.get(k) + value.getS()*value.getR());
					simMap.put(k, simMap.get(k) + value.getS());
				}
				else {
					sumMap.put(k, value.getS()*value.getR());
					simMap.put(k, value.getS());
				}
			}
			StringBuilder line = new StringBuilder("");
			for(long i=0;i<n;i++){
				if(i!=0) line.append(",");
				if(sumMap.containsKey(i))
					line.append(sumMap.get(i)/simMap.get(i));
				else line.append("0.0");
			}
			outValue.set(line.toString());
			context.write(key, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			n = context.getConfiguration().getInt("model.n", 1024);
			sumMap = new HashMap<Long, Double>(n);
			simMap = new HashMap<Long, Double>(n);
		}
	}
	
	/**
	 * Reducer for computing the prediction results and sorting
	 * @author tigerzhong
	 *
	 */
	public static class SortedComputeReducer extends Reducer<LongWritable, TripletWritable, LongWritable, Text> {
		
		/**
		 * Value for output
		 */
		private Text outValue = new Text();
		
		private HashMap<Long, Double> sumMap = null;
		
		private HashMap<Long, Double> simMap = null;
		
		private mapCMP comparator = new mapCMP();
		
		@Override
		public void reduce(LongWritable key, Iterable<TripletWritable> values, Context context)
				throws IOException, InterruptedException {
			for(TripletWritable value : values){
				long k = value.getP();
				if (sumMap.containsKey(k)){
					sumMap.put(k, sumMap.get(k) + value.getS()*value.getR());
					simMap.put(k, simMap.get(k) + value.getS());
				}
				else {
					sumMap.put(k, value.getS()*value.getR());
					simMap.put(k, value.getS());
				}
			}
			for(Map.Entry<Long, Double> e : simMap.entrySet())
				sumMap.put(e.getKey(), sumMap.get(e.getKey())/e.getValue());
			List<Map.Entry<Long, Double>> infoIds = new ArrayList<Map.Entry<Long, Double>>(
					sumMap.entrySet());
			Collections.sort(infoIds, comparator);
			StringBuilder line = new StringBuilder("");
			boolean first = true;
			for (Map.Entry<Long, Double> pair : infoIds) {
				if (!first)
					line.append(";");
				line.append(pair.getKey().toString());
				line.append(",");
				line.append(pair.getValue().toString());
				first = false;
			}
			outValue.set(line.toString());
			context.write(key, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			int n = context.getConfiguration().getInt("model.n", 1024);
			sumMap = new HashMap<Long, Double>(n);
			simMap = new HashMap<Long, Double>(n);
		}
	}
	
	public static class mapCMP implements Comparator<Map.Entry<Long, Double>> {

		@Override
		public int compare(Entry<Long, Double> first, Entry<Long, Double> second) {
			return first.getValue().compareTo(second.getValue());
		}
	}
}
