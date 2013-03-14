package com.tencent.bi.cf.model.rankmf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
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
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Class to generate pairwise information
 * u_i, v_j, r_j; u_i, v_k, r_k
 * ==> u_i, v_j, v_k | if r_j>r_k
 * ==> u_i, v_k, v_j | if r_j<r_k
 * @author tigerzhong
 *
 */
public class PairwiseGenerator {
	
	public static void getPariwise(String inputPath, String outputPath) throws IOException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(PairwiseGenerator.class);
		job.setJobName("MF-Generate-Pairwise");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PairwiseMapper.class);
		job.setReducerClass(PairwiseReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
	}

	/**
	 * Mapper for Pairwise
	 * @author tigerzhong
	 *
	 */
	public static class PairwiseMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",",2);
			outKey.set(items[0]);
			outText.set(items[1]);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * Reducer for Pairwise
	 * @author tigerzhong
	 *
	 */
	public static class PairwiseReducer extends Reducer<Text, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			outKey.set(key);
			Iterator<Text> it = values.iterator();
			List<String> idList = new ArrayList<String>();
			List<Double> valList = new ArrayList<Double>();
			while(it.hasNext()){
				String line = it.next().toString();
				String[] items = line.split(",");
				idList.add(items[0]);
				valList.add(Double.parseDouble(items[1]));
			}
			for(int i=0;i<idList.size();i++){
				for(int j=i;j<idList.size();j++){
					if(valList.get(i)>valList.get(j)){
						outText.set(idList.get(i)+","+idList.get(j));
						context.write(outKey, outText);
					}
					else if(valList.get(i)<valList.get(j)){
						outText.set(idList.get(j)+","+idList.get(i));	
						context.write(outKey, outText);
					}
				}		
			}
		}
		
	}
}
