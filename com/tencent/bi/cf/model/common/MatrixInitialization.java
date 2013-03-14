package com.tencent.bi.cf.model.common;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;



/**
 * Initialize a Matrix Distributed Randomly
 * @author tigerzhong
 *
 */
public class MatrixInitialization {
	/**
	 * Number of columns
	 */
	private int numCol = 0;
	/**
	 * Output path
	 */
	private String outputName = "";
	
	private String inputName = "";
	
	private int idx = 0;
	/**
	 * Initialize 
	 * @param numRow, number of rows
	 * @param numCol, number of columns
	 * @param inputName, number of input path
	 * @param pathName, path name
	 * @param gList, whether generating a list
	 * @throws IOException
	 */
	public void init(int numCol, int idx, String inputName, String pathName, boolean gList) throws IOException{
		this.numCol = numCol;
		this.outputName = pathName;
		this.inputName = inputName;
		this.idx = idx;
	}
	
	/**
	 * Perform generating
	 * @throws IOException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public void perform(Configuration conf) throws IOException, InterruptedException, ClassNotFoundException{
		conf.setInt("matrix.numcol", this.numCol);
		conf.setInt("matrix.idx", this.idx);
		conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(MatrixInitialization.class);
		job.setJobName("Matrix-Init");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(InitMapper.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputName));
		FileOutputFormat.setOutputPath(job, new Path(outputName));
		job.waitForCompletion(true);
	}
	
	/**
	 * Initialize mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class InitMapper extends Mapper<LongWritable, Text, Text, Text> {
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		private Text outKey = new Text();
		
		private int idx = 0;
		
		private int numD = 0;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			outKey.set(items[idx]);
			StringBuilder line = new StringBuilder();
			for(int i=0;i<numD;i++){
				if(i!=0) line.append(",");
				line.append(Math.random()/numD);
			}
			outText.set(line.toString());
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getInt("matrix.idx", 0);
			numD = context.getConfiguration().getInt("matrix.numcol", 10);
		}
	}

}
