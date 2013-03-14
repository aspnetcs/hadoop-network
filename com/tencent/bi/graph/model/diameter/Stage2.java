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

public class Stage2 {
	
	public static void perform(String inputPath, String outputPath) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(Stage2.class);
		job.setJobName("HADI-Stage2");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(IntWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(IntWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(Stage2Reducer.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));		
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	public static class Stage2Reducer extends Reducer<LongWritable, IntWritable, LongWritable, IntWritable> {
		
		private IntWritable outValue = new IntWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
			int bitmap = 0;
			for(IntWritable value : values) bitmap |= value.get();
			outValue.set(bitmap);
			context.write(key, outValue);
		}
	}
}
