package com.tencent.bi.cf.model.common;

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
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.biasmf.BiasMF;
import com.tencent.bi.cf.model.tensor.AbstractTensor;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;

/**
 * Prediction for MFs which store U and V in the main memory
 * @author tigerzhong
 *
 */
public class MFPrediction {
	/**
	 * Model
	 */
	protected static AbstractMF model = null;
	
	/**
	 * Prediction method
	 * @param inputPath, input path for pair data
	 * @param outputPath, output path for prediction results
	 * @param modelName, model name
	 * @throws Exception
	 */
	public static void predictAll(String inputPath, String modelName) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		conf.set("model.name", modelName);
		Job job = new Job(conf);
		job.setJarByClass(AbstractMF.class);
		job.setJobName("MF-Prediction-"+modelName);		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(DoubleWritable.class);
		job.setMapperClass(PredictionMapper.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(0);	//no reduce process
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"prediction-"+modelName+"/"));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		if(modelName.indexOf("Bias")>-1){	//for bias models
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"uBias.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"vBias.dat"), job.getConfiguration());
		} else if(modelName.indexOf("Tensor")>-1 || modelName.indexOf("CMF")>-1){	//for tensor and collaborative MF
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"S.dat"), job.getConfiguration());
		} else if(modelName.indexOf("RLMF")>-1){
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"B.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fu.dat"), job.getConfiguration());
			DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fv.dat"), job.getConfiguration());
		}
		job.waitForCompletion(true);
	}
	
	/**
	 * Mapper for prediction
	 * @author tigerzhong
	 *
	 */
	public static class PredictionMapper extends Mapper<LongWritable, Text, Text, DoubleWritable>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			//Pair
			String items[] = value.toString().split(",",-1);
			int p = Integer.parseInt(items[0]);
			int q = Integer.parseInt(items[1]);
			int o = -1;
			if(items.length>3)	//tensor or three matrices
				o = Integer.parseInt(items[2]);
			double r = model.predict(p,q,o);	//rating
			if(items.length>3) outKey.set(p+","+q+","+o);
			else outKey.set(p+","+q);
			outValue.set(r);
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			Configuration conf = context.getConfiguration();
			String modelName = conf.get("model.name");
			try {
				model = (AbstractMF) Class.forName(modelName).newInstance();
			} catch (Exception e) {
				e.printStackTrace();
			}
			//Read side data
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			model.setU(MatrixIO.readDenseMatrixFromText(paths[0].toString()));
			model.setV(MatrixIO.readDenseMatrixFromText(paths[1].toString()));
			if(modelName.indexOf("Bias")>-1){	//bias model
				BiasMF tmpModel = (BiasMF) model;
				tmpModel.setuBias((DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[2].toString()).viewColumn(0));
				tmpModel.setvBias((DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[3].toString()).viewColumn(0));
				model = tmpModel;
			}
			else if(modelName.indexOf("Tensor")>-1 || modelName.indexOf("CMF")>-1){	//tensor
				AbstractTensor tmpModel = (AbstractTensor) model;
				tmpModel.setS(MatrixIO.readDenseMatrixFromText(paths[2].toString()));
			}
		}
	}
}
