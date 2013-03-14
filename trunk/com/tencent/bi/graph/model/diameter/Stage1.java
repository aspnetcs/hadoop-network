package com.tencent.bi.graph.model.diameter;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.LongPairWritable;

public class Stage1 {
	
	public static void perform(String relationPath, String bitsPath, String outputPath) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(Stage1.class);
		job.setJobName("HADI-Stage1");	
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(IntWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setMapperClass(Stage1Mapper.class);
		job.setReducerClass(Stage1Reducer.class);
		FileInputFormat.addInputPath(job, new Path(relationPath));	
		FileInputFormat.addInputPath(job, new Path(bitsPath));	
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	public static class Stage1Mapper extends Mapper<LongWritable, Writable, LongPairWritable, LongWritable> {
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, Writable value, Context context)
				throws IOException, InterruptedException {
			if(value instanceof LongWritable){	//relational data
				outKey.set(((LongWritable) value).get(), 1);
				context.write(outKey, key);
			} else {
				outKey.set(key.get(), 0);
				outValue.set(((IntWritable)value).get());
				context.write(outKey, outValue);
			}
		}
	}

	public static class Stage1Reducer extends Reducer<LongPairWritable, LongWritable, LongWritable, IntWritable> {
		
		private IntWritable outValue = new IntWritable();
		
		private LongWritable outKey = new LongWritable();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			boolean flag = false;
			for(LongWritable value : values){
				if(key.getSecond()==0){
					outValue.set((int) value.get());
					flag = true;
				} else {
					if(!flag) break; 
					outKey.set(value.get());
					context.write(outKey, outValue);
				}
			}
			outKey.set(key.getFirst());
			context.write(outKey, outValue);
		}
	}
}
