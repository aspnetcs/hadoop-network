package com.tencent.bi.cf.model.common;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
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

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Get User and Item Bias, assuming the number of user is huge while the number of items is small
 * @author tigerzhong
 *
 */
public class BiasGenerator {
	
	/**
	 * Get the biases of users and items
	 * @param inputPath, input path for data
	 * @throws IOException
	 * @throws URISyntaxException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public static void getBias(String inputPath, int m, int n) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(BiasGenerator.class);
		job.setJobName("MF-UserItem-Bias");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(BiasMapper.class);
		job.setReducerClass(BiasReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"BiasTmp/"));
		job.waitForCompletion(true);
		//Postprocess
		extractBias(job.getConfiguration(), conf.get("hadoop.tmp.path")+"BiasTmp/", conf.get("hadoop.cache.path")+"vBias.dat", n);
		FileSystem fs = FileSystem.get(conf);
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"BiasTmp/"), true);
	}
	
	/**
	 * Combine bias data from small files to a integrated file
	 * @param conf
	 * @param pathName
	 * @param outName
	 * @throws IOException
	 */
	public static void extractBias(Configuration conf, String pathName, String outName, int n) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		DenseDoubleMatrix1D vBias = new DenseDoubleMatrix1D(n);
		for(String line : resList){
			String[] items = line.split("\t");
			vBias.set(Integer.parseInt(items[0]), Double.parseDouble(items[1]));
		}
		MatrixIO.saveDenseMatrix1D2HDFS(conf, outName, vBias, true);
	}
	
	/**
	 * Mapper for bias
	 * @author tigerzhong
	 *
	 */
	public static class BiasMapper extends Mapper<LongWritable, Text, Text, Text>{
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
			String[] items = value.toString().split(",");
			outKey.set("u,"+items[0]);
			outText.set(items[2]);
			context.write(outKey, outText);
			outKey.set("v,"+items[1]);
			outText.set(items[2]);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * Reducer for bias
	 * @author tigerzhong
	 *
	 */
	public static class BiasReducer extends Reducer<Text, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		/**
		 * Output Stream for uBias
		 */
		protected FSDataOutputStream uBiasFW = null;
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			double avg = 0.0, cnt = 0.0;
			Iterator<Text> it = values.iterator();
			while(it.hasNext()){
				avg += Double.parseDouble(it.next().toString());
				cnt += 1.0;
			}
			String[] items = key.toString().split(",");
			if(key.toString().startsWith("u")){
				//uBiasFW.write((items[1]+","+(avg/cnt)+"\n").getBytes());
				uBiasFW.write((items[1]+",0.0\n").getBytes());	//Since user's click is rare, please ignore the user bias!!
			}
			else{
				outKey.set(items[1]);
				outText.set(avg/cnt+"");
				context.write(outKey, outText);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf); 
			uBiasFW = fs.create(new Path(conf.get("hadoop.cache.path")+"uBias/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uBiasFW.close();
		}
	}
}
