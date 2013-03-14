package com.tencent.bi.cf.optimization.gradient.tensor;

import java.io.IOException;
import java.net.URI;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.gradient.common.CombinerReducer;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * SGD for Tensor Factorization
 * @author tigerzhong
 *
 */
public class SGDTensor implements TensorSolver {
	
	/**
	 * Loss function for tensor
	 */
	protected static TensorLoss lossFunc = null;
	/**
	 * Latent matrix for users
	 */
	protected static DenseDoubleMatrix2D U = null;
	/**
	 * Latent matrix for items
	 */
	protected static DenseDoubleMatrix2D V = null;
	/**
	 * Latent matrix for contexts
	 */
	protected static DenseDoubleMatrix2D S = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning Rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of Latent Dimension
	 */
	protected static int numD = 10;
		
	@Override
	public void initialize(Loss lossFunc, double lambda,
			double learningRate, int numD) throws Exception {
		SGDTensor.lossFunc = (TensorLoss) lossFunc;
		SGDTensor.lambda = lambda;
		SGDTensor.learningRate = learningRate;
		SGDTensor.numD = numD;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			DenseDoubleMatrix2D S, String inputPath, String outputPath) throws Exception {
		//Pre-Process, write U, V to HDFS as side-data
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"S.dat", S);
		conf.set("model.loss", ModelOperator.loss2String(conf, SGDTensor.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		// MR, get gradient
		Job job = new Job(conf);
		job.setJarByClass(SGDTensor.class);
		job.setJobName("Tensor-SGD-Update");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setNumReduceTasks(numD*10);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		DistributedCache.addCacheFile(new URI(catchPath+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"S.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		// Post-Process, update U, V
		extractUVS(SGDTensor.U, SGDTensor.V, SGDTensor.S, job.getConfiguration(), outputPath);

	}

	private void extractUVS(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D S, Configuration conf, String pathName) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("U")){
				for(int i=0;i<vals.length;i++)
					U.set(t, i, U.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			}
			else if(c.equalsIgnoreCase("V")){
				for(int i=0;i<vals.length;i++)
					V.set(t, i, V.get(t, i) - learningRate*Double.parseDouble(vals[i]));				
			}
			else if(c.equalsIgnoreCase("S")){
				for(int i=0;i<vals.length;i++)
					S.set(t, i, S.get(t, i) - learningRate*Double.parseDouble(vals[i]));				
			}
		}
	}


	public class SGDMapper extends Mapper<LongWritable, Text, Text, Text>{

		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outValue = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			//Triplet
			String items[] = value.toString().split(",",-1);
			int p = Integer.parseInt(items[1]);
			int q = Integer.parseInt(items[2]);
			int o = Integer.parseInt(items[3]);
			double r = Double.parseDouble(items[4]);
			DoubleMatrix1D a = null, b = null, c = null;
			a = lossFunc.getGradient(U.viewRow(p), V.viewRow(q), S.viewRow(o), r, lambda);
			b = lossFunc.getGradient(V.viewRow(q), U.viewRow(p), S.viewRow(o), r, lambda);
			c = lossFunc.getGradient(S.viewRow(o), U.viewRow(p), V.viewRow(q), r, lambda);
			
			long cnt = (long)(lossFunc.getValue(U.viewRow(p), V.viewRow(q), S.viewRow(o), r)*1000);
			context.getCounter("Eval", "Loss").increment(cnt);
			context.getCounter("Eval", "Cnt").increment(1);
			
			outKey.set("U,"+p);
			outValue.set(StringUtils.array2String(a.toArray()));
			context.write(outKey, outValue);
			outKey.set("V,"+q);
			outValue.set(StringUtils.array2String(b.toArray()));
			context.write(outKey, outValue);
			outKey.set("S,"+q);
			outValue.set(StringUtils.array2String(c.toArray()));
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			S = MatrixIO.readDenseMatrixFromText(paths[2].toString());
			SGDTensor.numD = conf.getInt("model.numD", 10);
			SGDTensor.lambda = conf.getFloat("model.lambda", 0.005f);
			SGDTensor.lossFunc = (TensorLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}

//////////////////////////////////////////////////////////////////////////////////////	
	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		// Ignore!!
		return;
	}
}
