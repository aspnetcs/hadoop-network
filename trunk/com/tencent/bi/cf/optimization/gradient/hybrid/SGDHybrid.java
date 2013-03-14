package com.tencent.bi.cf.optimization.gradient.hybrid;

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
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * SGD for hybrid MF
 * @author tigerzhong
 *
 */
public class SGDHybrid implements MFSolver{
	/**
	 * Regularization term
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
	/**
	 * Loss function
	 */
	protected static SquareHybridLoss lossFunc = null;
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		SGDHybrid.lossFunc = (SquareHybridLoss) lossFunc;
		SGDHybrid.lambda = lambda;
		SGDHybrid.learningRate = learningRate;
		SGDHybrid.numD = numD;
	}
	
	/**
	 * Update method
	 * @param U, latent matrix for user
	 * @param V, latent matrix for item
	 * @param B, regression matrix
	 * @param inputPath, input path for data
	 * @param outputPath, output path for training results
	 * @throws Exception
	 */
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D B,
			String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"U.dat", U);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"B.dat", B);
		conf.set("model.loss", ModelOperator.loss2String(conf, SGDHybrid.lossFunc));
		conf.setInt("model.numD", numD);	
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		//MR, get gradients
		Job job = new Job(conf);
		job.setJarByClass(SGDHybrid.class);
		job.setJobName("MFHybrid-SGD-Update");		
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
		DistributedCache.addCacheFile(new URI(catchPath+"B.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fu.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fv.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Post-Process, update U, V and B
		extract(U, V, B, job.getConfiguration(), outputPath);
	}
	
	/**
	 * Extract U, V and B
	 * @param U, latent matrix for user
	 * @param V, latent matrix for item
	 * @param B, regression matrix
	 * @param conf, configuration
	 * @param pathName, path name of data
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, DenseDoubleMatrix2D B, Configuration conf, String pathName) throws IOException{
		List<String> resList = DataOperators.readTextFromHDFS(conf, pathName);	//read all data from HDFS
		for(String line : resList){
			String[] items = line.split("\t");
			String c = items[0].split(",")[0];
			int t = Integer.parseInt(items[0].split(",")[1]);
			String[] vals = items[1].split(",");
			if(c.equalsIgnoreCase("U"))
				for(int i=0;i<vals.length;i++)
					U.set(t, i, U.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			else if(c.equalsIgnoreCase("V"))
				for(int i=0;i<vals.length;i++)
					V.set(t, i, V.get(t, i) - learningRate*Double.parseDouble(vals[i]));
			else if(c.equalsIgnoreCase("B")){
				int pt = 0;
				int numFu = Integer.parseInt(items[0].split(",")[1]);
				int numFv = Integer.parseInt(items[0].split(",")[2]);
				for(int i=0;i<numFu;i++)
					for(int j=0;j<numFv;j++)
						B.setQuick(i, j, B.get(i, j)- learningRate*Double.parseDouble(vals[pt++]));
			}
		}
	}
	
	/**
	 * Mapper for SGD
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Latent matrix for user
		 */
		protected DenseDoubleMatrix2D U = null;
		/**
		 * Latent matrix for item
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Regression matrix
		 */
		protected DenseDoubleMatrix2D B = null;
		/**
		 * Feature matrix for user
		 */
		protected DenseDoubleMatrix2D Fu = null;
		/**
		 * Feature matrix for item
		 */
		protected DenseDoubleMatrix2D Fv = null;
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output Text
		 */
		private Text outText = new Text();
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			//Triplet
			String items[] = value.toString().split(",",-1);
			int p = Integer.parseInt(items[0]);
			int q = Integer.parseInt(items[1]);
			double r = Double.parseDouble(items[2]);
			DoubleMatrix1D u = lossFunc.getGradient(U.viewRow(p), V.viewRow(q), B, Fu.viewRow(p), Fv.viewRow(q), r, lambda);
			DoubleMatrix1D v = lossFunc.getGradient(V.viewRow(q), U.viewRow(p), B, Fu.viewRow(p), Fv.viewRow(q), r, lambda);
			DoubleMatrix2D gB = lossFunc.getGradientB(V.viewRow(q), U.viewRow(p), B, Fu.viewRow(p), Fv.viewRow(q), r, lambda);
			long cnt = (long)(lossFunc.getValue(U.viewRow(p), V.viewRow(q), B, Fu.viewRow(p), Fv.viewRow(q), r)*1000);
			context.getCounter("Eval", "Loss").increment(cnt);
			context.getCounter("Eval", "Cnt").increment(1);
			outKey.set("U,"+p);
			outText.set(StringUtils.array2String(u.toArray()));
			context.write(outKey, outText);
			outKey.set("V,"+q);
			outText.set(StringUtils.array2String(v.toArray()));
			context.write(outKey, outText);
			outKey.set("B,"+gB.rows()+","+gB.columns()+","+q);
			outText.set(StringUtils.matrix2String(gB));
			context.write(outKey, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			U = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			V = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			B = MatrixIO.readDenseMatrixFromText(paths[2].toString());
			Fu = MatrixIO.readDenseMatrixFromText(paths[3].toString());
			Fv = MatrixIO.readDenseMatrixFromText(paths[4].toString());
			SGDHybrid.numD = conf.getInt("model.numD", 10);
			SGDHybrid.lambda = conf.getFloat("model.lambda", 0.005f);
			SGDHybrid.lossFunc = (SquareHybridLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
		}
	}
	

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	
	@Deprecated
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		//Ignore!!
	}

}
