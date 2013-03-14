package com.tencent.bi.cf.optimization.gradient.bias;

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
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombinerReducer;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Matrix Factorization with Bias
 * @author tigerzhong
 *
 */
public class SGDBias implements MFSolver {
	/**
	 * Loss function
	 */
	protected static SquareBiasLoss lossFunc = null;

	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of latent dimensions
	 */
	protected static int numD = 10;

	public void initialize(Loss lossFunc, double lambda, double learningRate, int numD)
			throws Exception {
		SGDBias.lossFunc = (SquareBiasLoss) lossFunc;
		SGDBias.lambda = lambda;
		SGDBias.learningRate = learningRate;
		SGDBias.numD = numD;	
	}

	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			DenseDoubleMatrix1D uBias, DenseDoubleMatrix1D vBias, String inputPath, String outputPath) throws Exception {
		//Pre-Process, write U, V, uBias, vBias to HDFS as side-data
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		MatrixIO.saveDenseMatrix1D2HDFS(conf, catchPath+"uBias.dat", uBias, true);
		MatrixIO.saveDenseMatrix1D2HDFS(conf, catchPath+"vBias.dat", vBias, true);
		conf.set("model.loss", ModelOperator.loss2String(conf, SGDBias.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		// MR, get gradient
		Job job = new Job(conf);
		job.setJarByClass(SGDBias.class);
		job.setJobName("MFBias-SGD-Update");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(CombinerReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		DistributedCache.addCacheFile(new URI(catchPath+"U.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"uBias.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"vBias.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		// Post-Process, update U, V
		extractUV(U, V, uBias, vBias, job.getConfiguration(), outputPath);

	}
	/**
	 * Extract gradients from file and update U, V, ubias and vbias
	 * @param U
	 * @param V
	 * @param uBias
	 * @param vBias
	 * @param conf
	 * @param pathName
	 * @throws IOException
	 */
	private void extractUV(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			DoubleMatrix1D uBias, DoubleMatrix1D vBias,
			Configuration conf, String pathName) throws IOException {
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);
		for (String line : resList) {
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if (c.equals("U")) {
				for (int i = 0; i < vals.length; i++)
					U.set(t, i, U.get(t, i) - learningRate * Double.parseDouble(vals[i]));
			} else if (c.equals("V")) {
				for (int i = 0; i < vals.length; i++)
					V.set(t, i, V.get(t, i) - learningRate * Double.parseDouble(vals[i]));
			} else if (c.equals("u")) {
				uBias.set(t, uBias.get(t) - learningRate * Double.parseDouble(vals[0]));
			} else if (c.equals("v")) {
				vBias.set(t, vBias.get(t) - learningRate * Double.parseDouble(vals[0]));
			}
			
		}
	}

	public static class SGDMapper extends
			Mapper<LongWritable, Text, Text, Text> {
		/**
		 * Latent Matrix for Users
		 */
		protected static DenseDoubleMatrix2D U = null;
		/**
		 * Latent Matrix for Items
		 */
		protected static DenseDoubleMatrix2D V = null;
		/**
		 * User Bias
		 */
		protected static DenseDoubleMatrix1D uBias = null;
		/**
		 * Item Bias
		 */
		protected static DenseDoubleMatrix1D vBias = null;
		/**
		 * Output key
		 */
		private static Text outKey = new Text();
		/**
		 * Output value
		 */
		private static Text outText = new Text();
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			// Triplet
			String items[] = value.toString().split(",", -1);
			int p = Integer.parseInt(items[0]);
			int q = Integer.parseInt(items[1]);
			double r = Double.parseDouble(items[2]);
			//Compute Gradient
			DoubleMatrix1D u = lossFunc.getGradient(U.viewRow(p), V.viewRow(q),
					uBias.get(p), vBias.get(q), r, lambda);
			DoubleMatrix1D v = lossFunc.getGradient(V.viewRow(q), U.viewRow(p),
					uBias.get(p), vBias.get(q), r, lambda);
			double ug = lossFunc.getGradientBias(V.viewRow(q), U.viewRow(p),
					uBias.get(p), vBias.get(q), r, lambda);
			double vg = lossFunc.getGradientBias(V.viewRow(q), U.viewRow(p),
					vBias.get(q), uBias.get(p), r, lambda);
			long cnt = (long)(lossFunc.getValue(U.viewRow(p), V.viewRow(q), vBias.get(q), uBias.get(p), r)*1000);
			context.getCounter("Eval", "Loss").increment(cnt);
			context.getCounter("Eval", "Cnt").increment(1);
			//Output
			outKey.set("U," + p);
			outText.set(StringUtils.array2String(u.toArray()));
			context.write(outKey, outText);
			outKey.set("V," + q);
			outText.set(StringUtils.array2String(v.toArray()));
			context.write(outKey, outText);
			outKey.set("u," + p);
			outText.set(ug+"");
			context.write(outKey, outText);
			outKey.set("v," + q);
			outText.set(vg+"");
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V, uBias, vBias
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			uBias = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[2].toString()).viewColumn(0);
			vBias = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[3].toString()).viewColumn(0);
			SGDBias.numD = conf.getInt("model.numD", 10);
			SGDBias.lambda = conf.getFloat("model.lambda", 0.005f);
			SGDBias.lossFunc = (SquareBiasLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		// TODO Auto-generated method stub
		
	}

}
