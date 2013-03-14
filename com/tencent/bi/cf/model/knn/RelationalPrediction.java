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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
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

import com.tencent.bi.cf.model.CF;
import com.tencent.bi.graph.utility.NumberLinksGenerator;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.LongPairSecondaryComparator;
import com.tencent.bi.utils.serialization.KeyPairWritable;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.TripletWritable;

/**
 * Predict the interaction using the relationship between users r_ij =
 * sum_c\in{N(i)}(r_cj)/M
 * 
 * @author tigerzhong
 * 
 */
public class RelationalPrediction implements CF {
	/**
	 * Path of relation data
	 */
	protected String relationPath;
	/**
	 * Path of interaction data
	 */
	protected String interPath;
	/**
	 * Number of items
	 */
	protected int n;
	/**
	 * Number of iteration
	 */
	protected int numIt;
	/**
	 * Trade-off parameter
	 */
	protected double alpha;
	
	/**
	 * Initialize model
	 * @param relationPath, path of relational data, such as qq-friend, microblog-follow, etc.
	 * @param interPath, path of interaction data, such as user-ad, user-vip, etc.
	 * @param numItem, number of items 
	 */
	public void initModel(String relationPath, String interPath, int numItem, int numIt, double alpha) {
		this.relationPath = relationPath;
		this.interPath = interPath;
		this.numIt = numIt;
		this.n = numItem;
		this.alpha = alpha;
	}

	@Override
	public void buildModel() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		//Get the edge number of each node
		NumberLinksGenerator.getNumLinks(this.relationPath, conf.get("hadoop.tmp.path")+"NumLinks/", true);
		performRandomWalk(0, conf);
		for(int i=1;i<numIt;i++){
			this.interPath = conf.get("hadoop.output.path") + "RelationalTrain/"+(i-1)+"/";
			performRandomWalk(i, conf);
		}
	}

	protected void performRandomWalk(int it, Configuration conf) throws Exception{
		conf.setInt("model.n", n);
		conf.setFloat("model.alpha", (float)alpha);
		conf.set("mapred.textoutputformat.separator", ",");
		conf.set("relation.path",this.relationPath);
		// First M/R, joining the followees and items
		Job job = new Job(conf);
		job.setJobName("Relational-Training-Join-"+it);
		job.setJarByClass(RelationalPrediction.class);
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(KeyPairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(TripletWritable.class);
		job.setMapperClass(HashJoinMapper.class);
		job.setReducerClass(HashJoinReducer.class);
		job.setPartitionerClass(FirstPartitioner.class);
		job.setSortComparatorClass(LongPairSecondaryComparator.KeyComparator.class);
		job.setGroupingComparatorClass(LongPairSecondaryComparator.GroupComparator.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(relationPath));
		FileInputFormat.addInputPath(job, new Path(interPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"NumLinks/"));
		FileOutputFormat
				.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")
						+ "RelationalPrediction/"));
		job.waitForCompletion(true);
		// Second M/R, counting the frequency of items
		job = new Job(conf);
		job.setJobName("Relational-Training-Count-"+it);
		job.setJarByClass(RelationalPrediction.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(KeyPairWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(CountMapper.class);
		job.setCombinerClass(CountCominer.class);
		job.setReducerClass(CountReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(conf.get("hadoop.tmp.path")
				+ "RelationalPrediction/"));
		FileOutputFormat.setOutputPath(job,
				new Path(conf.get("hadoop.output.path") + "RelationalTrain/"+it+"/"));
		job.waitForCompletion(true);
		// Postprocess
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get("hadoop.tmp.path") + "RelationalPrediction/"), true);
	}
	
	/*
	 * Mapper for joining relation data and interaction data
	 * @author tigerzhong
	 *
	 */
	public static class HashJoinMapper extends
			Mapper<LongWritable, Text, LongPairWritable, KeyPairWritable> {

		private LongPairWritable outKey = new LongPairWritable();

		private KeyPairWritable outValue = new KeyPairWritable();

		private String relationPath = "";
	
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String line = value.toString();
			if (path.indexOf(relationPath)>-1) { // relation, which is split by \t
				String[] pair = line.split("\t");
				outKey.set(Long.parseLong(pair[1]), 2);
				outValue.set(Long.parseLong(pair[0]),1.0);
			} else if (path.indexOf("NumLinks")>-1){
				String[] items = line.split(",");
				outKey.set(Long.parseLong(items[0]), 1);
				outValue.set(Long.parseLong(items[1]),0.0);
			} else { 				   // interaction, which is split by ,
				String[] items = line.split(",");
				outKey.set(Long.parseLong(items[0]), 0);
				outValue.set(Long.parseLong(items[1]),Double.parseDouble(items[2]));
			}
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			relationPath = context.getConfiguration().get("relation.path","");
		}
	}

	/**
	 * Reducer for joining relation data and interaction data 
	 * @author tigerzhong
	 *
	 */
	public static class HashJoinReducer extends
			Reducer<LongPairWritable, KeyPairWritable, LongWritable, TripletWritable> {

		private List<Long> keyList = null;
		
		private List<Double> valueList = null;

		private LongWritable outKey = new LongWritable();
		
		private TripletWritable outValue = new TripletWritable();
		
		private double alpha = 0.0;
		
		private int n = 1;
		
		@Override
		public void reduce(LongPairWritable key, Iterable<KeyPairWritable> values,
				Context context) throws IOException, InterruptedException {
			keyList.clear(); valueList.clear();
			boolean flagInter = false, flagNL = false;
			long numLinks = 0;
			for (KeyPairWritable value : values) {
				if (key.getSecond() == 0){	//load the intersection data first, since it is small
					keyList.add(value.getKey());
					valueList.add(value.getValue());
					flagInter = true ;
				} else if (key.getSecond() == 1) {
					numLinks = value.getKey();
					flagNL = true;
				} else {									//process the relation data one by one
					if(!(flagInter && flagNL)) break;		//No intersection data, so break!
					outKey.set(value.getKey());
					for (int i = 0; i < keyList.size(); i++) {
						outValue.set(keyList.get(i), value.getValue(), (1-alpha)*valueList.get(i)/numLinks);
						context.write(outKey, outValue);
					}
				}
			}
			outKey.set(key.getFirst());
			for (int i = 0; i < keyList.size(); i++) {
				outValue.set(keyList.get(i), 1.0, alpha/n);
				context.write(outKey, outValue);
			}
		}

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			n = context.getConfiguration().getInt("model.n", 1024);
			keyList = new ArrayList<Long>(n);
			valueList = new ArrayList<Double>(n);
			alpha = context.getConfiguration().getFloat("model.alpha", 0.0f);
		}
	}

	public static class FirstPartitioner extends Partitioner<LongPairWritable, KeyPairWritable> {

		HashPartitioner<Long, KeyPairWritable> p = new HashPartitioner<Long, KeyPairWritable>();
		
		@Override
		public int getPartition(LongPairWritable key, KeyPairWritable value,
				int numPartitions) {
			//return Math.abs((int) key.getFirst() * 127) % numPartitions;
			return p.getPartition(key.getFirst(), value, numPartitions);
		}

	}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////

	/**
	 * Mapper for computing the number of interaction between friends and items 
	 * @author tigerzhong
	 *
	 */	
	public static class CountMapper extends Mapper<LongWritable, TripletWritable, LongWritable, KeyPairWritable> {

		private KeyPairWritable outValue = new KeyPairWritable();

		@Override
		public void map(LongWritable key, TripletWritable value, Context context)
				throws IOException, InterruptedException {
			outValue.set(value.getP(), value.getR()*value.getS());
			context.write(key, outValue);
			//outValue.set(-1, value.getS());
			//context.write(key, outValue);
		}
	}

	public static class CountCominer extends Reducer<LongWritable, KeyPairWritable, LongWritable, KeyPairWritable> {

		private KeyPairWritable outValue = new KeyPairWritable();

		private HashMap<Long, Double> cntMap = null;

		@Override
		public void reduce(LongWritable key, Iterable<KeyPairWritable> values,
				Context context) throws IOException, InterruptedException {
			cntMap.clear();
			for (KeyPairWritable value : values) {
				long k = value.getKey();
				if (cntMap.containsKey(k)) cntMap.put(k, cntMap.get(k) + value.getValue());
				else cntMap.put(k, value.getValue());
			}
			for (Map.Entry<Long, Double> ent : cntMap.entrySet()) {
				outValue.set(ent.getKey(), ent.getValue());
				context.write(key, outValue);
			}
		}

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			int n = context.getConfiguration().getInt("model.n", 1024);
			cntMap = new HashMap<Long, Double>(n);
		}
	}

	public static class CountReducer extends Reducer<LongWritable, KeyPairWritable, LongWritable, Text> {
	
		private Text outValue = new Text();
		
		private HashMap<Long, Double> cntMap = null;

		@Override
		public void reduce(LongWritable key, Iterable<KeyPairWritable> values,
			Context context) throws IOException, InterruptedException {
			cntMap.clear();
			double sum = 1.0;
			for (KeyPairWritable value : values) {
				long k = value.getKey();
				//if(k==-1) {
				//	sum+=value.getValue();
				//	continue;
				//}
				if (cntMap.containsKey(k)) cntMap.put(k, cntMap.get(k) + value.getKey());
				else cntMap.put(k, value.getValue());
			}
			for(Map.Entry<Long, Double> e: cntMap.entrySet()){
				String outLine = e.getKey()+","+e.getValue()/sum;
				outValue.set(outLine);
				context.write(key, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {
			super.setup(context);
			int n = context.getConfiguration().getInt("model.n", 1024);
			cntMap = new HashMap<Long, Double>(n);
		}
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////
	public static class SortedCountReducer extends Reducer<LongWritable, KeyPairWritable, LongWritable, Text> {

		private Text outValue = new Text();

		private HashMap<Long, Double> cntMap = null;

		private mapCMP comparator = new mapCMP();

		@Override
		public void reduce(LongWritable key, Iterable<KeyPairWritable> values,
				Context context) throws IOException, InterruptedException {
			cntMap.clear();
			for (KeyPairWritable value : values) {
				long k = value.getKey();
				if(k==-1) { continue; }
				if (cntMap.containsKey(k)) cntMap.put(k, cntMap.get(k) + value.getValue());
				else cntMap.put(k, value.getValue());
			}
			List<Map.Entry<Long, Double>> infoIds = new ArrayList<Map.Entry<Long, Double>>(
					cntMap.entrySet());
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
			cntMap = new HashMap<Long, Double>(n);
		}
	}

	public static class mapCMP implements Comparator<Map.Entry<Long, Double>> {

		@Override
		public int compare(Entry<Long, Double> first, Entry<Long, Double> second) {
			return first.getValue().compareTo(second.getValue());
		}
	}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Override
	public void predictPair(String inputPath, int numD) throws Exception {

	}

	@Override
	public void predictAll() throws Exception {

	}

}
