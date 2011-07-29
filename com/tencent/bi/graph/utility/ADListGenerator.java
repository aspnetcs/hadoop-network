package com.tencent.bi.graph.utility;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
//import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class ADListGenerator {
		
	/**
	 * Generate adjacency list
	 */
	public static void generateADList(int numNode, int idx, String inputPath, String outputPath) throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("mapred.task.timeout", 6000000);
		conf.setInt("graph.numNode", numNode);
		conf.setInt("line.idx", idx);
		Job job = new Job(conf);
		job.setJarByClass(ADListGenerator.class);
		job.setJobName("RW-AdjacencyList-Generation");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(ADMapper.class);
		job.setReducerClass(ADReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}
	
	/**
	 * Mapper for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class ADMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{

		LongWritable outKey = new LongWritable();
		
		LongWritable outValue = new LongWritable();
		
		int idx = 0;
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(Long.parseLong(items[idx]));
			outValue.set(Long.parseLong(items[1-idx]));
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getInt("line.idx", 1);
		}
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class ADReducer extends Reducer<LongWritable, LongWritable, LongWritable, MatrixRowWritable>{
				
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private List<Long> ids = new ArrayList<Long>();
		
		private List<Double> adList = new ArrayList<Double>();
		
		private int numNode = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			for(LongWritable item : values){
				ids.add(item.get());
				adList.add(1.0);
			}
			outValue.set(ids, adList, numNode);
			context.getCounter("Eval", "Cnt").increment(1);
	  		context.write(key, outValue);
	  		ids.clear(); adList.clear();
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			numNode = context.getConfiguration().getInt("graph.numNode", 1);
		}
	}
}
