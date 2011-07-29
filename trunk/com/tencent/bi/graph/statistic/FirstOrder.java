package com.tencent.bi.graph.statistic;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;

public class FirstOrder {
	
	public static void getNumEdges(String inputPath) throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		//First MR, distributing data
		Job job = new Job(conf);
		job.setJarByClass(FirstOrder.class);
		job.setJobName("FirstOrder-Number-Edges");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(CNTMapper.class);
		job.setCombinerClass(CNTReducer.class);
		job.setReducerClass(CNTReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"NumOfEdges/"));
		job.waitForCompletion(true);
	}

	public static class CNTMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{

		LongWritable outKey = new LongWritable();
		
		LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			outKey.set(0);
			outValue.set(1);
			context.getCounter("Eval", "Cnt").increment(1);
			context.write(outKey, outValue);
		}
	}
	
	public static class CNTReducer extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable>{
		
		private LongWritable outValue = new LongWritable();

		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			long cnt = 0;			
			for(LongWritable item : values){
				cnt += item.get();
			}
			outValue.set(cnt);
	  		context.write(key, outValue);
		}
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public static void getFollowDistribution(String inputPath, int idx) throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("data.idx", idx);
		//First MR, 
		Job job = new Job(conf);
		job.setJarByClass(FirstOrder.class);
		job.setJobName("FirstOrder-User-Count");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(HashMapper.class);
		job.setCombinerClass(CNTReducer.class);
		job.setReducerClass(CNTReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"UserCount/"));
		job.waitForCompletion(true);
		//Second MR
		conf.setInt("data.idx", 1);
		job = new Job(conf);
		job.setJarByClass(FirstOrder.class);
		job.setJobName("FirstOrder-User-Distribution");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(HashMapper.class);
		job.setCombinerClass(CNTReducer.class);
		job.setReducerClass(CNTReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"UserCount/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"FollowDistribution-"+idx+"/"));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"UserCount/"), true);
	}
	
	public static class HashMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{

		private LongWritable outKey = new LongWritable();
		
		private LongWritable outValue = new LongWritable();
		
		private int idx = 0;
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(Long.parseLong(items[idx]));
			outValue.set(1);
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getInt("data.idx", 1);
		}
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
