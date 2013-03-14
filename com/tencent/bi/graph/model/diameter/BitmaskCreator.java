package com.tencent.bi.graph.model.diameter;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;

public class BitmaskCreator {
	
	protected String inputPath;
	
	protected String outputPath;
	
	protected final static int K = 32;
	
	public static void create(String inputPath, String outputPath) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(BitmaskCreator.class);
		job.setJobName("BitmaskCreator");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(IntWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setMapperClass(CreatorMapper.class);
		job.setReducerClass(CreatorReducer.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));	
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	public static class CreatorMapper extends Mapper<LongWritable, LongWritable, LongWritable, LongWritable> {
		
		@Override
		public void map(LongWritable key, LongWritable value, Context context)
				throws IOException, InterruptedException {
			context.write(value, key);
		}
	}
	
	public static class CreatorReducer extends Reducer<LongWritable, LongWritable, LongWritable, IntWritable> {
		
		private IntWritable outValue = new IntWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			int bitmap = 0;
			for(LongWritable value : values){
				Long v = value.get();
				Integer hash = v.hashCode();
				int id = K-1;
				if(hash!=0) id = Integer.numberOfTrailingZeros(hash);
				bitmap |= (1<<id);
			}
			outValue.set(bitmap);
			context.write(key, outValue);
		}
	}
	
}
