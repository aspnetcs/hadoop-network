package com.tencent.bi.topic.model.plsa;

import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
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
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.LongPairWritable;

@Deprecated
public class CoOccurrenceModel extends PLSA {

	@Override
	public void buildModel() throws Exception {
		for(int i=0;i<this.numIt;i++){
			//E-step
			Configuration conf = FileOperators.getConfiguration();
			conf.set("mapred.textoutputformat.separator", ",");
			String catchPath = conf.get("hadoop.cache.path");
			conf.setInt("model.numZ", z);
			MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"pzw.dat", pzw);
			MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"pzd.dat", pzd);
			MatrixIO.saveDenseMatrix1D2HDFS(conf, catchPath+"pz.dat", pz, true);
			Job job = new Job(conf);
			job.setJarByClass(CoOccurrenceModel.class);
			job.setJobName("PLSA-C-EStep-"+i);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			job.setMapperClass(EMapper.class);
			job.setNumReduceTasks(0);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.setInputPaths(job, new Path(inputPath));
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"PLSA-E/"));
			DistributedCache.addCacheFile(new URI(catchPath+"pzd.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(catchPath+"pzw.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(catchPath+"pz.dat"), job.getConfiguration());
			job.waitForCompletion(true);
			//M-step
			job = new Job(conf);
			job.setJarByClass(CoOccurrenceModel.class);
			job.setJobName("PLSA-C-MStep-D-"+i);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			job.setMapperClass(MDisMapper.class);
			job.setReducerClass(MDisReducer.class);
			job.setNumReduceTasks(conf.getInt("mapred.reduce.tasks", 100));
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(inputPath));
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"PLSA-E/"));
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"PLSA-M/"));
			DistributedCache.addCacheFile(new URI(catchPath+"pzd.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(catchPath+"pzw.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(catchPath+"pz.dat"), job.getConfiguration());
			job.waitForCompletion(true);
			job = new Job(conf);
			job.setJarByClass(CoOccurrenceModel.class);
			job.setJobName("PLSA-C-MStep-C-"+i);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(Text.class);
			job.setMapperClass(CombineMapper.class);
			job.setReducerClass(CombineReducer.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"PLSA-M/"));
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"PLSA-M/"));
			job.waitForCompletion(true);
		}
	}

	public static class EMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
		/**
		 * P(d|z)
		 */
		protected DenseDoubleMatrix2D pzd = null;
		/**
		 * P(w|z)
		 */
		protected DenseDoubleMatrix2D pzw = null;
		/**
		 * P(z)
		 */
		protected DenseDoubleMatrix1D pz = null;
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		protected int numZ = 0;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Triplet
			String[] items = value.toString().split(",");
			double[] pdwz = new double[numZ];
			int d = Integer.parseInt(items[0]);
			int w = Integer.parseInt(items[1]);
			double sum = 0.0;
			for(int i=0;i<numZ;i++){
				pdwz[i] = Double.parseDouble(items[2])*pz.get(i)*pzd.get(i, d)*pzw.get(i, w);
				sum += pdwz[i];
			}
			for(int i=0;i<numZ;i++){
				outKey.set(items[0]+","+items[1]+","+i);
				outValue.set(pdwz[i]);
				context.write(outKey, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			pzd = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			pzw = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			pz = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[2].toString()).viewColumn(0);
			numZ = conf.getInt("model.numZ", 10);
		}
	}
	
	public static class MDisMapper extends Mapper<LongWritable, Text, LongPairWritable, Text> {
		/**
		 * Output key
		 */
		private LongPairWritable outKey = new LongPairWritable();
		/**
		 * Output value
		 */
		private Text outValue = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Triplet
			String[] items = value.toString().split(",",3);
			outKey.set(Long.parseLong(items[0]), Long.parseLong(items[1]));
			outValue.set(items[2]);
			context.write(outKey, outValue);
		}

	}
	
	public static class MDisReducer extends Reducer<LongPairWritable, Text, Text, DoubleWritable> {
		/**
		 * P(d|z)
		 */
		protected DenseDoubleMatrix2D pzd = null;
		/**
		 * P(w|z)
		 */
		protected DenseDoubleMatrix2D pzw = null;
		/**
		 * P(z)
		 */
		protected DenseDoubleMatrix1D pz = null;
		
		protected int numZ = 0;
		
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<Text> values,
				Context context) throws IOException, InterruptedException {
			double[] z = new double[numZ];
			double cnt = 0.0;
			for(Text value: values){
				String[] items = value.toString().split(",");
				if(items.length<2) cnt = Double.parseDouble(items[0]);
				else {
					z[Integer.parseInt(items[0])] = Double.parseDouble(items[1]);
				}
			}
			for(int i=0;i<numZ;i++){
				outKey.set("d,"+i+","+key.getFirst());
				outValue.set(cnt*z[i]);
				context.write(outKey, outValue);
				outKey.set("w,"+i+","+key.getSecond());
				outValue.set(cnt*z[i]);
				context.write(outKey, outValue);
				outKey.set("z,"+i+","+(int)(Math.random()*context.getNumReduceTasks()));
				outValue.set(cnt*z[i]);
				context.write(outKey, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			pzd = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			pzw = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			pz = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[2].toString()).viewColumn(0);
			numZ = conf.getInt("model.numZ", 10);
		}
	}
	
	public static class CombineMapper extends Mapper<LongWritable, Text, Text, DoubleWritable> {
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Triplet
			String[] items = value.toString().split(",");
			outKey.set(items[0]+","+items[1]+","+items[2]);
			outValue.set(Double.parseDouble(items[3]));
			context.write(outKey, outValue);
		}

	}
	
	public static class CombineReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void reduce(Text key, Iterable<DoubleWritable> values,
				Context context) throws IOException, InterruptedException {
			double r = 0.0;
			for(DoubleWritable value : values) r += value.get();
			outValue.set(r);
			context.write(key, outValue);
		}
	}
}
