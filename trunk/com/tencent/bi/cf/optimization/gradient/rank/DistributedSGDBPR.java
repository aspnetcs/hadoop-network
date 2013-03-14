package com.tencent.bi.cf.optimization.gradient.rank;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
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
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.PlusMult;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombineCombiner;
import com.tencent.bi.cf.optimization.gradient.common.CombineReducer;
import com.tencent.bi.cf.optimization.gradient.common.Extraction;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPairPartitioner;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPartitioner;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Distributed SGD for BPR-MF
 * @author tigerzhong
 *
 */
public class DistributedSGDBPR implements MFSolver {
	/**
	 * Loss function
	 */
	protected static AUCLoss lossFunc = null;
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
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		DistributedSGDBPR.lossFunc = (AUCLoss) lossFunc;
		DistributedSGDBPR.lambda = lambda;
		DistributedSGDBPR.learningRate = learningRate;
		DistributedSGDBPR.numD = numD;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		//Preprocess, write V to HDFS as side-data
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		String tmpPath = conf.get("hadoop.tmp.path")+"DistributedSGDBPR/";
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		conf.set("model.loss", ModelOperator.loss2String(conf, DistributedSGDBPR.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		
		//First MR, obtaining gradient
		Job job = new Job(conf);
		job.setJarByClass(DistributedSGDBPR.class);
		job.setJobName("BPRMF-DistributedSGD-Update-Phase1");		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(SGDMapper.class);
		job.setPartitionerClass(ModLongPartitioner.class);
		job.setReducerClass(SGDReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(catchPath+"U/"));
		FileOutputFormat.setOutputPath(job, new Path(tmpPath));
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		
		//Clear
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(catchPath+"U/"), true);
		
		//Second MR, update V
		job = new Job(conf);
		job.setJarByClass(DistributedSGDBPR.class);
		job.setJobName("BPRMF-DistributedSGD-Update-Phase2");
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(CombineMapper.class);
		job.setCombinerClass(CombineCombiner.class);
		job.setPartitionerClass(ModLongPairPartitioner.class);
		job.setReducerClass(CombineReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(tmpPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
		
		//Postprocess
		Extraction.extract(V, job.getConfiguration(), outputPath, learningRate);
		fs.delete(new Path(tmpPath), true);
	}
	
	/**
	 * Mapper for SGD, distributing data
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable>{
		/**
		 * Output key
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			outKey.set(Long.parseLong(items[0])/100);
			if(path.indexOf("/U")>-1){	//latent factors for user
				double[] vec = new double[items.length-1];
				for(int i=0;i<items.length-1;i++)
					vec[i] = Double.parseDouble(items[i+1]);
				outValue.set(vec);
			} else {					//data
				if(items.length!=3) return;
				outValue.set(Long.parseLong(items[1]), Long.parseLong(items[2]));
			}
			context.write(outKey, outValue);
		}
	}
	
	/**
	 * Reducer for SGD, computing the gradients
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<LongWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Latent vector for the current user
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Latent Matrix for Item
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Output Key
		 */
		private LongPairWritable outKey = new LongPairWritable();
		/**
		 * Output Text
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context)
				throws IOException, InterruptedException {
			List<Integer> aList = new ArrayList<Integer>(1024);
			List<Integer> bList = new ArrayList<Integer>(1024);
			for(MatrixRowWritable value : values){
				if (!value.isSparse()){	//U
					U = (DenseDoubleMatrix1D) value.getDenseVector().copy();			
				} else{ 								//Data
					aList.add((int) value.getFirstID());
					bList.add((int) value.getFirstVal());
				}
			}
			int numPair = aList.size();
			for(int i=0;i<numPair;i++){
				int q1 = aList.get(i);
				int q2 = bList.get(i);
				//Get gradient
				DoubleMatrix1D u = lossFunc.getGradient(U, V.viewRow(q1), V.viewRow(q2), 1, lambda);	
				DoubleMatrix1D v1 = lossFunc.getGradient(U, V.viewRow(q1), V.viewRow(q2), 2, lambda);
				DoubleMatrix1D v2 = lossFunc.getGradient(U, V.viewRow(q1), V.viewRow(q2), 3, lambda);	
				//Get loss
				long cnt = (long)(lossFunc.getValue(U, V.viewRow(q1), V.viewRow(q2))*1000);
				context.getCounter("Eval", "Loss").increment(cnt);
				context.getCounter("Eval", "Cnt").increment(1);
				//Output V and update U
				outKey.set(q1,0);
				outValue.set(v1.toArray());
				context.write(outKey, outValue);
				outKey.set(q2,0);
				outValue.set(v2.toArray());
				context.write(outKey, outValue);
				U.assign(u, PlusMult.plusMult(-DistributedSGDBPR.learningRate/numPair));
			}
			//Output
			outKey.set(key.get(), 1);
			outValue.set(U.toArray());
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			Path[] paths = DistributedCache.getLocalCacheFiles(conf); //Read V from side data
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			DistributedSGDBPR.numD = conf.getInt("model.numD", 10);
			DistributedSGDBPR.lambda = conf.getFloat("model.lambda", 0.005f);
			DistributedSGDBPR.lossFunc = (AUCLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
			U = new DenseDoubleMatrix1D(DistributedSGDBPR.numD);
		}
	}
	
	/**
	 * Second mapper, distributing V while outputting U
	 * @author tigerzhong
	 *
	 */
	public static class CombineMapper extends Mapper<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output stream for U
		 */
		protected FSDataOutputStream uFW = null;
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		@Override
		public void map(LongPairWritable key, MatrixRowWritable value, Context context)
				throws IOException, InterruptedException {
			if(key.getSecond()==1){
				uFW.write((key.getFirst()+","+StringUtils.array2String(value.getVector())+"\n").getBytes());
			} else {
				//Others
				double[] vec = value.viewVector();
				double[] outVec = new double[vec.length+1];
				outVec[0] = 1.0;	//count
				for(int i=0;i<vec.length;i++) outVec[i+1] = vec[i];
				outValue.set(outVec);
				context.write(key, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf); 
			uFW = fs.create(new Path(conf.get("hadoop.cache.path")+"U/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uFW.close();
		}
	}
}
