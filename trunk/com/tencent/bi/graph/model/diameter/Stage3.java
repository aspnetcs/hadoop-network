package com.tencent.bi.graph.model.diameter;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
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

public class Stage3 {
	
	public static void perform(String inputPath, String outputPath) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(Stage3.class);
		job.setJobName("HADI-Stage3");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(DoubleWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(DoubleWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setMapperClass(Stage3Mapper.class);
		job.setCombinerClass(Stage3Reducer.class);
		job.setReducerClass(Stage3Reducer.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));	
		FileInputFormat.addInputPath(job, new Path(outputPath));	
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	public static class Stage3Mapper extends Mapper<LongWritable, IntWritable, IntWritable, FloatWritable> {
		
		private FloatWritable outValue = new FloatWritable();
		
		private IntWritable outKey = new IntWritable(0);
		
		@Override
		public void map(LongWritable key, IntWritable value, Context context)
				throws IOException, InterruptedException {
			int b = Integer.numberOfLeadingZeros(~value.get());
			outValue.set((float)((1<<b)/0.77351));
			context.write(outKey, outValue);
		}
	}

	public static class Stage3Reducer extends Reducer<IntWritable, FloatWritable, IntWritable, FloatWritable> {
		
		private FloatWritable outValue = new FloatWritable();
		
		@Override
		public void reduce(IntWritable key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {
			float outV = 0.0f;
			for(FloatWritable value : values){
				outV += value.get();
			}
			context.write(key, outValue);
		}
	}
}
