package com.tencent.bi.graph.utility;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;

public class NumberLinksGenerator {
	
	public static void getNumLinks(String inputPath, String outputPath)  throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = FileOperators.getConfiguration();
		conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(ImportanceGenerator.class);
		job.setJobName("Number-Links-Generator");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(DoubleWritable.class);
		job.setMapperClass(ADListGenerator.ADMapper.class);
		job.setReducerClass(TMPReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);	
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class TMPReducer extends Reducer<LongWritable, LongWritable, LongWritable, DoubleWritable>{
				
		private DoubleWritable outValue = new DoubleWritable();
		
		@SuppressWarnings("unused")
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			int numLinks = 0;
			for(LongWritable v : values) numLinks++;
			outValue.set(numLinks);
	  		context.write(key, outValue);
	  		context.getCounter("Eval", "Cnt").increment(1);
		}
		
	}
}
