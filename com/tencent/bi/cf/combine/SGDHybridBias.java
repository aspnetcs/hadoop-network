package com.tencent.bi.cf.combine;

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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.ModelOperator;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;


/**
 * Distributed SGD for HybridBiasMF
 * @author tigerzhong
 *
 */
public class SGDHybridBias {
	/**
	 * Loss function
	 */
	protected Loss lossFunc = null;
	/**
	 * Regularization parameter
	 */
	protected double lambda = 0.005;
	/**
	 * Learning rate
	 */
	protected double learningRate = 0.01;
	/**
	 * Number of latent dimension
	 */
	protected int numD = 10;
	
	protected double ru = 0.0;
	
	protected double ri = 0.0;
	
	protected double rf = 0.0;
	
	public void initialize(Loss lossFunc, double lambda, double learningRate, double a, double b, double c, int numD) throws Exception {
		this.lossFunc = lossFunc;
		this.lambda = lambda;
		this.learningRate = learningRate;
		this.numD = numD;
		this.ru = a;
		this.ri = b;
		this.rf = c;
	}
	
	public void update(double[][] U, double[][] V, double[][] B, double[] uBias, double[] vBias,
			String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		String cachePath = conf.get("hadoop.cache.path");
		
		//Initializing the configuration
		conf.set("model.loss", ModelOperator.loss2String(conf, this.lossFunc));				//loss function
		conf.setInt("model.numD", numD);													//number of latent dimension
		conf.setFloat("model.learningRate", (float) learningRate);							//learning rate
		conf.setFloat("model.lambda", (float) lambda);										//regularization parameter
		conf.setFloat("model.a", (float) ru);
		conf.setFloat("model.b", (float) ri);
		conf.setFloat("model.c", (float) rf);
		//Saving data for distributed cache	
		MatrixIO.saveDenseMatrix2D2HDFS(conf, cachePath+"V.dat", V);
		MatrixIO.saveDenseMatrix2D2HDFS(conf, cachePath+"B.dat", B);
		MatrixIO.saveDenseMatrix1D2HDFS(conf, cachePath+"vBias.dat", vBias, true);
		
		//First MR, getting gradients of U, V, uBias, vBias, B; updating U and uBias
		Job job = new Job(conf);
		job.setJarByClass(SGDHybridBias.class);
		job.setJobName("HyBiasMF-DistributedSGD-Update-Phase1");
		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(SGDReducer.class);
		
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		FileInputFormat.addInputPath(job, new Path(inputPath));			//multiple inputs
		FileInputFormat.addInputPath(job, new Path(cachePath+"U/"));
		FileInputFormat.addInputPath(job, new Path(cachePath+"uBias/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.data.path")+"Fu/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"DistributedSGDHybridBias/"));	//distributed cache
		
		DistributedCache.addCacheFile(new URI(cachePath+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(cachePath+"vBias.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fv.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(cachePath+"B.dat"), job.getConfiguration());
		job.waitForCompletion(true);
		
		//Postprocess, removing all temporary outputs
		FileSystem fs = FileSystem.get(conf); 			
		fs.delete(new Path(cachePath+"U"), true);
		fs.delete(new Path(cachePath+"uBias"), true);
		
		//Second MR, updating B, V and vBias
		job = new Job(conf);
		job.setJarByClass(SGDHybridBias.class);
		job.setJobName("HyBiasMF-DistributedSGD-Update-Phase2");
		
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongPairWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		
		job.setMapperClass(CombineMapper.class);
		job.setCombinerClass(CombineCombiner.class);
		job.setReducerClass(CombineReducer.class);
		
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		FileInputFormat.setInputPaths(job, new Path(conf.get("hadoop.tmp.path")+"DistributedSGDHybridBias/"));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);
		//Postprocess
		extract(V, vBias, B, conf, outputPath); //Extract B, V, vBias from temporary outputs
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"DistributedSGDHybridBias/"), true);			
	}

	private void extract(double[][] V, double[] vBias, double[][] B, Configuration conf, String pathName) throws IOException{
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
				if(key.getSecond()==0L){			//V
					for(int i=0;i<vec.length;i++)
						V[k][i] -= learningRate*vec[i];
				} else if(key.getSecond()==1L) {	//vBias
					vBias[k] -= learningRate*vec[0];
				} else if(key.getSecond()==4L) {
					int pt = 0;
					for(int i=0;i<B.length;i++)
						for(int j=0;j<B[i].length;j++)
							B[i][j] -= learningRate*vec[pt++];
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
		 * Output Key
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output Text
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();	//get the path of input data
			String[] items = value.toString().split(",");			//get key
			double[] vec = new double[items.length];
			outKey.set(Long.parseLong(items[0]));
			if(path.indexOf("/U")>-1) vec[0] = 0.0; 				//latent factors for user
			else if (path.indexOf("/uBias")>-1) vec[0] = 1.0;		//bias factors for user
			else if (path.indexOf("/Fu")>-1) vec[0] = 2.0; 			//features for user
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
	 * First reducer, getting the gradients of all factors and updating U and uBias
	 * @author tigerzhong
	 *
	 */
	public static class SGDReducer extends Reducer<LongWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Latent vector for the current user
		 */
		protected double[] U = null;
		/**
		 * Feature vector for the current user
		 */
		protected double[] Fu = null;
		/**
		 * Bias for the current user
		 */
		protected double uBias = 0;
		/**
		 * Regression matrix
		 */
		protected double[][] B = null;
		/**
		 * Gradients of B
		 */
		protected double[] gB = null;
		/**
		 * Latent matrix for items
		 */
		protected double[][] V = null;
		/**
		 * Feature matrix for items
		 */
		protected double[][] Fv = null;
		/**
		 * Bias matrix for items
		 */
		protected double[] vBias = null;
		
		protected Map<Integer, Double> valueList = null;
		
		protected double a = 0.0;
		
		protected double b = 0.0;
		
		protected double c = 0.0;
		
		protected double lambda = 0.0;
		
		protected int numD = 10;
		
		protected double learningRate = 0.1;
		
		protected HybridBiasLoss lossFunc = null;
		/**
		 * Output Key
		 */
		private LongPairWritable outKey = new LongPairWritable();
		/**
		 * Output Text
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			valueList.clear();
			for(int i=0;i<gB.length;i++) gB[i] = 0.0;
			for(MatrixRowWritable value : values){
				if(value.isSparse()){ //data
					valueList.put((int)value.getFirstID(), value.getFirstVal());
				} else {			  //
					double[] vec = value.getVector();
					if(vec[0]<0.5){ 		//latent factors for user
						for(int i=0;i<vec.length-1;i++)
							U[i] = vec[i+1];
					} else if (vec[0]<1.5){ //bias factors for user
						uBias = vec[1];
					} else if (vec[0]<2.5) {  //features for user
						for(int i=0;i<vec.length-1;i++)
							Fu[i] = vec[i+1];
					}
				}
			}
			int numR = valueList.size();		//number of data for the current user
			for(Map.Entry<Integer, Double> E : valueList.entrySet()){	//process each click data
				int q = E.getKey();
				double r = E.getValue();
				//Compute prediction
				double pre = lossFunc.getPrediction(U, V[q], B, Fu, Fv[q], vBias[q], uBias, a, b, c);
				//Compute square error, for monitor
				long cnt = (long)(lossFunc.getValue(pre, r)*1000);
				context.getCounter("Eval", "Loss").increment(cnt);
				context.getCounter("Eval", "Cnt").increment(1);
				//Get gradient
				double[] u = lossFunc.getGradient(U, V[q], pre, r, lambda);
				double[] v = lossFunc.getGradient(V[q], U, pre, r, lambda);
				double[][] gb = lossFunc.getGradientB(B, Fu, Fv[q], pre, r, lambda, c);
				double ub = lossFunc.getGradientBias(uBias, pre, r, lambda, a);
				double vb = lossFunc.getGradientBias(vBias[q], pre, r, lambda, b);
				//Output V and vBias's sub gradients
				outKey.set(q,0);
				outValue.set(v);
				context.write(outKey, outValue);
				outKey.set(q,1);
				outValue.set(vb);
				context.write(outKey, outValue);
				//Combine B's gradients
				int pt = 0;
				for(int i=0;i<gb.length;i++)
					for(int j=0;j<gb[i].length;j++) gB[pt++] += gb[i][j]/numR;
				//Update U and uBias
				for(int i=0;i<numD;i++)
					U[i] -= learningRate*u[i]/numR;
				uBias -= learningRate*ub/numR;
			}
			//Output the updated U and uBias
			outKey.set(key.get(), 2);
			outValue.set(U);
			context.write(outKey, outValue);
			outKey.set(key.get(), 3);
			outValue.set(uBias);
			context.write(outKey, outValue);
			outKey.set(key.get()%context.getNumReduceTasks(), 4);
			outValue.set(gB);
			context.write(outKey, outValue);
			//0, V; 1, vBias; 2, U; 3, uBias; 4, B
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read V, vBias, FV and B
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			V = MatrixIO.readDense2DArrayFromText(paths[0].toString());		//latent matrix for items
			vBias = MatrixIO.readDense1DArrayFromText(paths[1].toString());	//item bias
			Fv = MatrixIO.readDense2DArrayFromText(paths[2].toString());	//feature matrix for items
			B = MatrixIO.readDense2DArrayFromText(paths[3].toString());		//regression matrix
			numD = conf.getInt("model.numD", 10);							//number of latent dimension
			learningRate = conf.getFloat("model.learningRate", 0.005f);
			lambda = conf.getFloat("model.lambda", 0.005f);	//reguralization
			a = conf.getFloat("model.a", 1.0f);
			b = conf.getFloat("model.b", 1.0f);
			c = conf.getFloat("model.c", 1.0f);
			lossFunc = (HybridBiasLoss) ModelOperator.string2Loss(conf,conf.get("model.loss")); //loss function
			valueList = new HashMap<Integer, Double>(); 
			U = new double[numD];
			gB = new double[B.length*B[0].length];
			Fu = new double[B.length];
		}
	}
	
	/**
	 * Second mapper, for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class CombineMapper extends Mapper<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output stream for U
		 */
		private FSDataOutputStream uFW = null;
		/**
		 * Output stream for uBias
		 */
		private FSDataOutputStream uBiasFW = null;
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongPairWritable key, MatrixRowWritable value, Context context)
				throws IOException, InterruptedException {
			/*Output the U and uBias for specific path and distribute the gradient of V, vBias and B*/
			if(key.getSecond()==2l){	//U
				uFW.write((key.getFirst()+","+StringUtils.array2String(value.getVector())+"\n").getBytes());
			} else if(key.getSecond()==3l){	//uBias
				uBiasFW.write((key.getFirst()+","+value.getFirstVal()+"\n").getBytes());
			} else {	//Others, V, vBias and B
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
			//Open output stream
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf); 
			uFW = fs.create(new Path(conf.get("hadoop.cache.path")+"U/"+((long)(Math.random()*1000000000))+".dat"));
			uBiasFW = fs.create(new Path(conf.get("hadoop.cache.path")+"uBias/"+((long)(Math.random()*1000000000))+".dat"));
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			super.cleanup(context);
			uFW.close();
			uBiasFW.close();
		}
	}
	
	public static class CombineCombiner extends Reducer<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
				
		@Override
		public void reduce(LongPairWritable key, Iterable<MatrixRowWritable> values,
				Context context) throws IOException, InterruptedException {
			double[] res = null;
			for (MatrixRowWritable value: values) {	//combining
				double[] vec = value.viewVector();
				if(res==null) res = new double[vec.length];
				for (int i = 0; i < vec.length; i++)
					res[i] += vec[i];
			}
			outValue.set(res);
			context.write(key, outValue);
		}
	}
	
	public static class CombineReducer extends Reducer<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongPairWritable key, Iterable<MatrixRowWritable> values,
				Context context) throws IOException, InterruptedException {
			double[] res = null;
			double pt = 0.0;
			for (MatrixRowWritable value: values) {	//combining
				double[] vec = value.viewVector();
				if(res==null) res = new double[vec.length-1];
				for (int i = 1; i < vec.length; i++)
					res[i-1] += vec[i];
				pt += vec[0];
			}
			for(int i=0;i<res.length;i++)
				res[i] /= pt;
			outValue.set(res);
			context.write(key, outValue);
		}
	}

}
