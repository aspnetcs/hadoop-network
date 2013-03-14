package com.tencent.bi.graph.utility;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;

public class NumberLinksGenerator {
	
	public static void getNumLinks(String inputPath, String outputPath, boolean isText)  throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = FileOperators.getConfiguration();
		if(isText) {
			conf.set("mapred.textoutputformat.separator", ",");
		}
		conf.setInt("line.idx", 1);
		Job job = new Job(conf);
		job.setJarByClass(ImportanceGenerator.class);
		job.setJobName("Number-Links-Generator");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		if(isText){
			job.setMapperClass(TextTMPMapper.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
		} else {
			job.setMapperClass(TMPMapper.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			
		}
		job.setCombinerClass(TMPReducer.class);
		job.setReducerClass(TMPReducer.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);	
	}
	
	public static class TextTMPMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{
		
		LongWritable outValue = new LongWritable();
		
		LongWritable outKey = new LongWritable();
		
		int idx = 0;
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(Long.parseLong(items[idx]));
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getInt("line.idx", 1);
			outValue.set(1);
		}
	}
	
	public static class TMPMapper extends Mapper<LongWritable, LongWritable, LongWritable, LongWritable>{
		
		LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, LongWritable value, Context context) throws IOException, InterruptedException {
			outValue.set(1);
			context.write(key, outValue);
		}
		

	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class TMPReducer extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable>{
				
		private LongWritable outValue = new LongWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			int numLinks = 0;
			for(LongWritable v : values) numLinks+=v.get();
			outValue.set(numLinks);
	  		context.write(key, outValue);
	  		context.getCounter("Eval", "Cnt").increment(1);
		}
		
	}
}
