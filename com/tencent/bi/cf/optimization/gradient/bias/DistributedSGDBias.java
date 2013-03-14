package com.tencent.bi.cf.optimization.gradient.bias;

import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.cf.optimization.gradient.common.CombineCombiner;
import com.tencent.bi.cf.optimization.gradient.common.CombineReducer;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPairPartitioner;
import com.tencent.bi.cf.optimization.gradient.common.LongPartition.ModLongPartitioner;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.SeqBlas;

/**
 * Distributed SGD with Bias
 * @author tigerzhong
 *
 */
public class DistributedSGDBias implements MFSolver{
	/**
	 * Loss function
	 */
	protected static SquareBiasLoss lossFunc = null;
	/**
	 * Latent matrix for items
	 */
	protected static DenseDoubleMatrix2D V = null;
	/**
	 * Item bias
	 */
	protected static DenseDoubleMatrix1D vBias = null;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected static double learningRate = 0.01;
	/**
	 * Number of latent dimension
	 */
	protected static int numD = 10;
	
	public void initialize(Loss lossFunc, double lambda,
			double learningRate, int numD) throws Exception {
		DistributedSGDBias.lossFunc = (SquareBiasLoss) lossFunc;
		DistributedSGDBias.lambda = lambda;
		DistributedSGDBias.learningRate = learningRate;
		DistributedSGDBias.numD = numD;
	}

	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			DenseDoubleMatrix1D uBias, DenseDoubleMatrix1D vBias,
			String inputPath, String outputPath) throws Exception {
		//Initialization
		Configuration conf = FileOperators.getConfiguration();
		String catchPath = conf.get("hadoop.cache.path");
		String tmpPath = conf.get("hadoop.tmp.path")+"DistributedSGDBias/";
		conf.set("model.loss", ModelOperator.loss2String(conf, DistributedSGDBias.lossFunc));
		conf.setInt("model.numD", numD);
		conf.setFloat("model.learningRate", (float) learningRate);
		conf.setFloat("model.lambda", (float) lambda);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, catchPath+"V.dat", V);
		MatrixIO.saveDenseMatrix1D2HDFS(conf, catchPath+"vBias.dat", vBias, true);
		// MR, get gradient, update U and uBias
		Job job = new Job(conf);
		job.setJarByClass(DistributedSGDBias.class);
		job.setJobName("MFBias-DistributedSGD-Update-Phase1");
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
		FileInputFormat.addInputPath(job, new Path(catchPath+"U"));
		FileInputFormat.addInputPath(job, new Path(catchPath+"uBias"));
		FileOutputFormat.setOutputPath(job, new Path(tmpPath));
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"vBias.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(catchPath+"U"), true);
		fs.delete(new Path(catchPath+"uBias"), true);
		//MR, update V and Vbias
		job = new Job(conf);
		job.setJarByClass(DistributedSGDBias.class);
		job.setJobName("MFBias-DistributedSGD-Update-Phase2");
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
		DistributedCache.addCacheFile(new URI(catchPath+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(catchPath+"vBias.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		//Postprocess
		extract(V, vBias, new Configuration(), outputPath);
		fs.delete(new Path(tmpPath), true);	
	}
	
	/**
	 * Extract V and vBias from output file
	 * @param V
	 * @param vBias
	 * @param conf
	 * @param pathName
	 * @throws IOException
	 */
	private void extract(DenseDoubleMatrix2D V, DoubleMatrix1D vBias, Configuration conf, String pathName) throws IOException {
		//Update V and VBias
		FileSystem fs = FileSystem.get(conf); 
		FileStatus fsta[] = fs.globStatus(new Path(pathName+"*"));
		LongPairWritable key = new LongPairWritable();
		MatrixRowWritable value = new MatrixRowWritable();
		for (FileStatus it : fsta) {
			Path singlePath = it.getPath();
			if(it.isDir()) continue;
			SequenceFile.Reader rd = new SequenceFile.Reader(fs, singlePath, new Configuration());
			while(rd.next(key, value)){				//Processing line by line
				double[] vec = value.viewVector();
				int k = (int)key.getFirst();
				if(key.getSecond()==0){			//V
					for(int i=0;i<vec.length;i++){
						V.set(k,i, V.get(k, i) - learningRate*vec[i]);
					}
				} else if(key.getSecond()==1) {	//vBias
					vBias.set(k, vBias.get(k)-learningRate*vec[0]);
				}
			}
			rd.close();
		}
	}
	
	
	/**
	 * First mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class SGDMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable> {
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
			String[] items = value.toString().split(",");			//get key
			double[] vec = new double[items.length];
			outKey.set(Long.parseLong(items[0]));
			if(path.indexOf("/U")>-1) vec[0] = 0.0; 				//latent factors for user
			else if (path.indexOf("/uBias")>-1) vec[0] = 1.0;		//bias factors for user
			else {													//data
				if(items.length==3)
					outValue.set(Long.parseLong(items[1]), Double.parseDouble(items[2]));
				else outValue.set(Long.parseLong(items[1]), 0.0);
				context.write(outKey, outValue);
				return;
			}
			for(int i=1;i<items.length;i++)
				vec[i] = Double.parseDouble(items[i]);
			outValue.set(vec);
			context.write(outKey, outValue);
		}
	}
	
	/**
	 * First reducer, for computing the gradient of V, vBias and updating U, uBias
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<LongWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {

		/**
		 * Latent Matrix for the Current User
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Current User Bias
		 */
		protected double uBias = 0.0;
		/**
		 * Output key
		 */
		private LongPairWritable outKey = new LongPairWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		/**
		 * Value Map
		 */
		private Map<Integer, Double> valueList = null;
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values,
				Context context) throws IOException, InterruptedException {
			//Get Data
			valueList.clear();
			for(MatrixRowWritable value : values){
				if(value.isSparse()){ //data
					valueList.put((int)value.getFirstID(), value.getFirstVal());
				} else {			  //
					double[] vec = value.getVector();
					if(vec[0]<0.5){ 		//latent factors for user
						for(int i=0;i<vec.length-1;i++)
							U.set(i,vec[i+1]);
					} else if (vec[0]<1.5){ //bias factors for user
						uBias = vec[1];
					}
				}
			}
			int numR = valueList.size();
			//Compute Gradient, and update U and uBias at the same time
			for(Integer q : valueList.keySet()){
				double r = valueList.get(q);	//Rating
				DoubleMatrix1D u = lossFunc.getGradient(U, V.viewRow(q), uBias, vBias.get(q), r, lambda);	//U
				SeqBlas.seqBlas.daxpy(-learningRate/numR, u, U);
				DoubleMatrix1D v = lossFunc.getGradient(V.viewRow(q), U, uBias, vBias.get(q), r, lambda);	//V
				uBias -= learningRate/numR*lossFunc.getGradientBias(V.viewRow(q), U, uBias, vBias.get(q), r, lambda);			//UBias
				double vb = lossFunc.getGradientBias(V.viewRow(q), U, vBias.get(q), uBias, r, lambda);		//VBias
				/*Compute Square Error, for monitor*/
				long cnt = (long)(lossFunc.getValue(U, V.viewRow(q), vBias.get(q), uBias, r)*1000);
				context.getCounter("Eval", "Loss").increment(cnt);
				context.getCounter("Eval", "Cnt").increment(1);
				//Output the gradient of V, vBias
				outKey.set(q,0);
				outValue.set(v.toArray());
				context.write(outKey, outValue);
				outKey.set(q,1);
				outValue.set(vb);
				context.write(outKey, outValue);
			}
			//Output the new U and uBias
			outKey.set(key.get(), 2);
			outValue.set(U.toArray());
			context.write(outKey, outValue);
			outKey.set(key.get(), 3);
			outValue.set(uBias);
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V, vBias
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			DistributedSGDBias.numD = conf.getInt("model.numD", 10);
			DistributedSGDBias.lambda = conf.getFloat("model.lambda", 0.005f);
			DistributedSGDBias.lossFunc = (SquareBiasLoss) ModelOperator.string2Loss(conf,conf.get("model.loss"));
			U = new DenseDoubleMatrix1D(DistributedSGDBias.numD);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			vBias = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[1].toString()).viewColumn(0);
			valueList = new HashMap<Integer, Double>();
		}
	}
	
	/**
	 * Second mapper, distributing V and vBias while outputting U and uBias 
	 * @author tigerzhong
	 *
	 */
	public static class CombineMapper extends Mapper<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output stream for U
		 */
		protected FSDataOutputStream uFW = null;
		/**
		 * Output stream for uBias
		 */
		protected FSDataOutputStream uBiasFW = null;
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		@Override
		public void map(LongPairWritable key, MatrixRowWritable value, Context context)
				throws IOException, InterruptedException {
			if(key.getSecond()==2){
				uFW.write((key.getFirst()+","+StringUtils.array2String(value.getVector())+"\n").getBytes());
			} else if(key.getSecond()==3){
				uBiasFW.write((key.getFirst()+","+value.getFirstVal()+"\n").getBytes());
			} else {
				//Others, V, vBias and B
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
			DistributedSGDBias.numD = context.getConfiguration().getInt("model.numD", 10);	//number of latent dimension
			DistributedSGDBias.learningRate = context.getConfiguration().getFloat("model.learningRate", (float) 0.1);
			String catchPath = conf.get("hadoop.cache.path") ;
			uFW = fs.create(new Path(catchPath+"U/"+((long)(Math.random()*1000000000))+".dat"));
			uBiasFW = fs.create(new Path(catchPath+"uBias/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uFW.close();
			uBiasFW.close();
		}
	}

	@Deprecated
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V,
			String inputPath, String outputPath) throws Exception {
		
	}
	
}
