package com.tencent.bi.cf.model.knn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.DoublePairWritable;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

@Deprecated
public class KNNPredictionPair {


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

	public static void predict(String pairPath, String trainingPath, boolean userBase) throws Exception{
		Configuration conf = FileOperators.getConfiguration(); 
		conf.setBoolean("model.userBase", userBase);
		conf.set("mapred.textoutputformat.separator", ",");
		//Combine
		Job job = new Job(conf);
		job.setJobName("KNN-Prediction-Combine");
		job.setJarByClass(KNNPredictionPair.class);
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(CombineMapper.class);
		job.setReducerClass(CombineReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(pairPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"NgList/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNCombine/"));
		job.waitForCompletion(true);
		//Reverse
		job = new Job(conf);
		job.setJobName("KNN-Prediction-Reverse");
		job.setJarByClass(NgBuilder.class);
		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(DoublePairWritable.class);
		job.setMapperClass(ReverseMapper.class);
		job.setReducerClass(ReverseReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNCombine/"));
		FileInputFormat.setInputPaths(job, new Path(trainingPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNReverse/"));
		job.waitForCompletion(true);
		//Compute
		job = new Job(conf);
		job.setJobName("KNN-Prediction-Compute");
		job.setJarByClass(NgBuilder.class);
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(DoublePairWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(ComputeReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNReverse/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"KNNReverse/"));
		job.waitForCompletion(true);
	}
	
	public static class CombineMapper extends Mapper<LongWritable, Text, LongWritable, Text> {
		/**
		 * Key for output
		 */
		private LongWritable outKey = new LongWritable();
		
		/**
		 * Value for output
		 */
		private Text outValue = new Text();

		private boolean userBase = true;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",",2);
			if(path.indexOf("/NgList")>-1){
				if(userBase) {
					outKey.set(Long.parseLong(items[0]));
					outValue.set(items[1]);
				}
				else {
					outKey.set(Long.parseLong(items[1]));
					outValue.set(items[0]);
				}
			} else {
				outKey.set(Long.parseLong(items[0]));
				outValue.set(items[1]);
			}			
		}
		
		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			userBase = conf.getBoolean("model.userBase", true);
		}
	}
	
	public static class CombineReducer extends Reducer<LongWritable, Text, Text, MatrixRowWritable> {
		
		/**
		 * Value for output
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private Text outKey = new Text();
		
		protected List<String> idList = new ArrayList<String>();
		
		protected List<Double> valList = new ArrayList<Double>();
		
		protected List<String> tList = new ArrayList<String>();

		@Override
		public void reduce(LongWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			for(Text value : values){
				String[] items = value.toString().split(",");
				if(items.length>1){//NgList
					for(int i=0;i<items.length;i+=2){
						idList.add(items[i]);
						valList.add(Double.parseDouble(items[i+1]));
					}
				} else {
					tList.add(items[1]);
				}
			}
			for(String item: tList){
				for(int i=0;i<idList.size();i++){
					outKey.set(idList.get(i)+","+item);
					outValue.set(key.get(), valList.get(i));
					context.write(outKey, outValue);
				}
			}
		}
	}
	
	public static class ReverseMapper extends Mapper<LongWritable, Text, LongPairWritable, Text> {
		/**
		 * Key for output
		 */
		private LongPairWritable outKey = new LongPairWritable();
		
		/**
		 * Value for output
		 */
		private Text outValue = new Text();

		private boolean userBase = true;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",",3);
			if(path.indexOf("/KNNCombine")>-1){//similarity
				outKey.set(Long.parseLong(items[0]), Long.parseLong(items[1]));
				outValue.set(items[2]);
			} else {
				if(userBase) outKey.set(Long.parseLong(items[0]), Long.parseLong(items[1]));
				else outKey.set(Long.parseLong(items[1]), Long.parseLong(items[0]));
				outValue.set(items[2]);
			}			
		}
		
		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			userBase = conf.getBoolean("model.userBase", true);
		}
	}
	
	public static class ReverseReducer extends Reducer<LongPairWritable, Text, LongPairWritable, DoublePairWritable> {
		
		/**
		 * Value for output
		 */
		private DoublePairWritable outValue = new DoublePairWritable();
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private boolean userBase = true;
		
		private double defaultValue = 0.0;
		
		@Override
		public void reduce(LongPairWritable key, Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			double sim = 0.0, r = Double.NaN;
			long id = 0;
			for(Text value : values){
				String[] items = value.toString().split(",");
				if(items.length<2) r = Double.parseDouble(items[0]);
				else {
					id = Long.parseLong(items[0]);
					sim = Double.parseDouble(items[1]);
				}
			}
			if(Double.isNaN(r)) r=defaultValue;	//No rating!!
			if(userBase) outKey.set(id, key.getSecond());
			else outKey.set(key.getSecond(), id);
			outValue.set(sim, r);
			context.write(outKey, outValue);
		}
		
		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			userBase = conf.getBoolean("model.userBase", true);
			defaultValue = conf.getFloat("model.default", 0.0f);
		}
	}
	
	public static class ComputeReducer extends Reducer<LongPairWritable, DoublePairWritable, LongPairWritable, DoubleWritable> {
		
		/**
		 * Value for output
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<DoublePairWritable> values, Context context)
				throws IOException, InterruptedException {
			double simSum = 0.0, rSum = 0.0;
			for(DoublePairWritable value : values){
				simSum += value.getFirst();
				rSum += value.getFirst()*value.getSecond();
			}
			outValue.set(rSum/simSum);
			context.write(key, outValue);
		}

	}
}
