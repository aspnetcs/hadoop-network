package com.tencent.bi.graph.utility;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class ImportanceGenerator {
	
	/**
	 * Generate node importance values from click data
	 */
	public static void generateValueList(int numValues, int numNodes, boolean isADList, String inputPath, String outputPath)  throws IOException, InterruptedException, ClassNotFoundException {
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numValues", numValues);
		conf.setInt("model.numNodes", numNodes);
		if(!isADList) conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(ImportanceGenerator.class);
		job.setJobName("RW-Values-Generation");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(LongWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setMapperClass(ADListGenerator.ADMapper.class);
		if(isADList){
			job.setOutputValueClass(MatrixRowWritable.class);
			job.setReducerClass(TMPReducer.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
		}
		else {
			job.setOutputValueClass(DoubleWritable.class);
			job.setReducerClass(TMPTextReducer.class);
			job.setOutputFormatClass(TextOutputFormat.class);
		}
		job.setInputFormatClass(TextInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);		
	}

	/**
	 * Mapper for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class ValueMapper extends Mapper<LongWritable, Text, LongWritable, Text>{

		LongWritable outKey = new LongWritable();
		
		Text outValue = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split(",",2);
			outKey.set(Long.parseLong(items[0]));
			outValue.set(items[1]);
			context.write(outKey, outValue);
		}
		
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class ValueReducer extends Reducer<LongWritable, Text, LongWritable, MatrixRowWritable>{
				
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private int numValues = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			DenseDoubleMatrix1D valueList = new DenseDoubleMatrix1D(numValues);
			for(Text item : values){
				String[] items = item.toString().split(",");
				valueList.setQuick(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
			}
			outValue.set(valueList);
	  		context.write(key, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			numValues = context.getConfiguration().getInt("model.numValues", 1);
		}
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class TMPReducer extends Reducer<LongWritable, LongWritable, LongWritable, MatrixRowWritable>{
				
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private int numNode = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			outValue.set(1.0/numNode);
	  		context.write(key, outValue);
	  		context.getCounter("Eval", "Cnt").increment(1);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			numNode = context.getConfiguration().getInt("model.numNode", 1);
		}
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class TMPTextReducer extends Reducer<LongWritable, LongWritable, LongWritable, DoubleWritable>{
				
		private DoubleWritable outValue = new DoubleWritable();
		
		private int numNode = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			outValue.set(1.0/numNode);
	  		context.write(key, outValue);
	  		context.getCounter("Eval", "Cnt").increment(1);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			numNode = context.getConfiguration().getInt("model.numNode", 1);
		}
	}
}
