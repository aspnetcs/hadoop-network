package com.tencent.bi.cf.model.common;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

public class DistributedMFPrediction {
	
	public static void predictAll(String outputPath, String modelName) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		conf.set("mapred.textoutputformat.separator", ",");
		conf.set("model.name", modelName);
		//First MR, transforming dataset
		Job job = new Job(conf);
		job.setJarByClass(DistributedMFPrediction.class);
		job.setJobName("MF-Prediction-All");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(MultiplyMapper.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"MF-Prediction/"));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		job.waitForCompletion(true);
	}
	
	/**
	 * Mapper, distributing data
	 */
	public static class MultiplyMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		private DoubleMatrix2D V = null;
		
		private AbstractMF model = null;
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			DoubleMatrix1D u = new DenseDoubleMatrix1D(items.length-1);
			for(int i=1;i<items.length;i++){
				u.set(i-1, Double.parseDouble(items[i]));
			}
			StringBuilder outLine = new StringBuilder();
			for(int i=0;i<V.rows();i++){
				if(i!=0) outLine.append(",");
				outLine.append(model.predict(u, V.viewRow(i), null));
			}
			outKey.set(items[0]);
			outText.set(outLine.toString());
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			try {
				model = (AbstractMF) Class.forName(conf.get("model.name")).newInstance();
			} catch (InstantiationException e) {
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
		}
	}
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	
	public static void distributV() throws IOException{
		Configuration conf = FileOperators.getConfiguration();
		List<String> dataV = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"V.dat");
		FileSystem fs = FileSystem.get(new Configuration());
		FSDataOutputStream vFW = fs.create(new Path(conf.get("hadoop.cache.path")+"V/"+((long)(Math.random()*1000000000))+".dat")); 
		for(int i=0;i<dataV.size();i++){
			vFW.write((i+","+dataV.get(i)+'\n').getBytes());
		}
		vFW.close();
	}
	
	/**
	 * Perform prediction distributed using given pairs
	 * @param inputPath
	 * @param outputPath
	 * @param modelName
	 * @param numD
	 * @throws Exception
	 */
	public static void predictPair(String inputPath, String outputPath, String modelName, int numD) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numD", numD);
		conf.set("mapred.textoutputformat.separator", ",");
		conf.set("model.name", modelName);
		//First MR, transforming dataset
		Job job = new Job(conf);
		job.setJarByClass(DistributedMFPrediction.class);
		job.setJobName("DistMF-Prediction-Phase1");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(MatchMapper.class);
		job.setReducerClass(MatchReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"V"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"MF-Prediction/"));
		job.waitForCompletion(true);
		//Second MR, computing each u_i*v_j
		conf.set("mapred.textoutputformat.separator", "\t");
		conf.set("model.name", modelName);
		job = new Job(conf);
		job.setJarByClass(DistributedMFPrediction.class);
		job.setJobName("DistMF-Prediction-Phase2");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PredictMapper.class);
		job.setReducerClass(PredictReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"MF-Prediction"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath+"prediction-MF/"));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(job.getConfiguration());
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"MF-Prediction/"), true);	
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	 * First mapper, matching the row and distributing the data
	 */
	public static class MatchMapper extends Mapper<LongWritable, Text, Text, Text>{
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
			String[] items = value.toString().trim().split(",",2);
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			if(path.indexOf("/V")>-1){	//vector
				outKey.set(items[0]);
				outText.set(items[1]);
			} else { //data, item id as key
				String[] ids = items[1].split(",");
				outKey.set(ids[0]);
				outText.set(items[0]+","+ids[1]+",d");					
			}
			context.write(outKey, outText);
		}
	}
	/**
	 * First reducer, expanding the data to [i, r_ij, u_i] 
	 * @author tigerzhong
	 *
	 */
	public static class MatchReducer extends Reducer<Text, Text, Text, Text> {
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
			DoubleMatrix1D V = null;
			List<Integer> vList = new ArrayList<Integer>(); //value list
			Iterator<Text> it = values.iterator();
			while(it.hasNext()){
				String line = it.next().toString().trim();
				String[] items = line.split(",");
				if(items[items.length-1].equals("d")){ 	//Data
					vList.add(Integer.parseInt(items[0]));
				}
				else {									//V
					V = new DenseDoubleMatrix1D(items.length);
					for(int i=0;i<items.length;i++){
						V.set(i, Double.parseDouble(items[i]));
					}
				}
			}
			for(Integer k : vList){ //Output
				outKey.set(k+"");
				outText.set("d,"+key.toString()+","+StringUtils.array2String(V.toArray()));
				context.write(outKey, outText);
			}
		}
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	 * Second mapper, distributing data
	 */
	public static class PredictMapper extends Mapper<LongWritable, Text, Text, Text>{
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
	 * Second reducer, getting the prediction value
	 * @author tigerzhong
	 *
	 */
	public static class PredictReducer extends Reducer<Text, Text, Text, Text> {
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		private AbstractMF model = null;
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			DoubleMatrix1D U = null; //current user latent vector
			List<DoubleMatrix1D> V = new ArrayList<DoubleMatrix1D>(); //item latent matrix
			List<Integer> vidList = new ArrayList<Integer>();
			Iterator<Text> it = values.iterator();
			int p = Integer.parseInt(key.toString());
			while(it.hasNext()){			//saving
				String line = it.next().toString();
				String[] items = line.split(",");
				if(line.startsWith("d")){	//data plus V
					vidList.add(Integer.parseInt(items[1]));
					DoubleMatrix1D v = new DenseDoubleMatrix1D(items.length-2);
					for(int i=0;i<items.length-2;i++)
						v.set(i, Double.parseDouble(items[i+2]));
					V.add(v); 
				} else{
					U = new DenseDoubleMatrix1D(items.length);
					for(int i=0;i<items.length;i++)
						U.set(i, Double.parseDouble(items[i]));
				}
			}
			for(int i=0;i<vidList.size();i++){
				int q = vidList.get(i);
				DoubleMatrix1D v = V.get(i);
				double r = model.predict(U, v, null);
				//Output
				outKey.set(p+","+q);
				outText.set(r+"");
				context.write(outKey, outText);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			try {
				model = (AbstractMF) Class.forName(conf.get("model.name")).newInstance();
			} catch (InstantiationException e) {
				e.printStackTrace();
			} catch (IllegalAccessException e) {
				e.printStackTrace();
			} catch (ClassNotFoundException e) {
				e.printStackTrace();
			}
		}
	}
}
