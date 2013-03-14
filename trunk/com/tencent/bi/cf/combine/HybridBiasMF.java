package com.tencent.bi.cf.combine;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.combine.SGDHybridBias.SGDMapper;
import com.tencent.bi.cf.model.common.BiasGenerator;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

import cern.colt.matrix.DoubleFactory2D;
import cern.jet.math.Mult;

/**
 * HybridMF
 * Distributed HybridMF which decomposes the click matrix and learns a regression matrix
 * It explores the multiple inputs to handle huge user number
 * R = UV' + a*uBias + b*vBias + c*Fu*B*Fv
 * @author tigerzhong
 *
 */
public class HybridBiasMF {
	/**
	 * Latent matrix for items
	 */
	protected double[][] V = null;
	/**
	 * Regression matrix
	 */
	protected double[][] B = null;
	/**
	 * Item bias
	 */
	protected double[] vBias = null;
	
	/**
	 * Loss function
	 */
	protected HybridBiasLoss lossFunc = null;
	/**
	 * Optimization method
	 */
	protected SGDHybridBias solver = null;
	/**
	 * Number of iterations
	 */
	protected int numIt = 100;
	/**
	 * Learning rate
	 */
	protected double learningRate = 0.01;
	
	/**
	 * Number of latent factors
	 */
	protected int numD = 5;
	
	protected int numFv;
	
	protected int numFu;
	
	protected int numU;
	
	protected int numV;
	
	/**
	 * Regularization parameters
	 */
	protected double lambda = 0.005;
	
	protected double ru = 0.0;
	
	protected double ri = 0.0;
	
	protected double rf = 0.0;

	/**
	 * Data input path
	 */
	protected String inputPath = "";
	
	/**
	 * Initialize model
	 * @param m, number of users
	 * @param n, number of items
	 * @param d, number of latent dimensions
	 * @param fm, number of user features
	 * @param fn, number of item features
	 * @param solverName, optimization method
	 * @param lambda, reguralization parameter
	 * @param learningRate, learning rate
	 * @param numIt, number of iterations
	 * @param inputPath, input path for data
	 * @param outputPath, output path for training results
	 * @throws Exception
	 */
	public void initModel(int m, int n, int d, int fm, int fn, double a, double b, double c, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		
		this.learningRate = learningRate;
		this.numIt = numIt;
		solver = new SGDHybridBias();
		lossFunc = new HybridBiasLoss();
		
		this.numU = m;
		this.numV = n;
		this.numD = d;
		this.numFu = fm;
		this.numFv = fn;
		
		this.lambda = lambda;
		this.ru = a;
		this.ri = b;
		this.rf = c;
		
		this.inputPath = inputPath;
		
		B = DoubleFactory2D.dense.random(fm,fn).assign(Mult.div(fm)).toArray();
		V = DoubleFactory2D.dense.random(n,d).assign(Mult.div(d)).toArray();
		
		init(m, n, d);
	}
	
	
	public void buildModel() throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<numIt;i++){
			solver.initialize(lossFunc, lambda, learningRate, this.ru, this.ri, this.rf, numD);
			solver.update(null, V, B, null, vBias, inputPath, conf.get("hadoop.output.path")+"/DistributedHybridBiasMF/"+i+"/");
		}			
	}

	/**
	 * Initialize U, uBias and vBias
	 * @param m, number of users
	 * @param d, number of latent dimensions
	 * @throws IOException
	 * @throws ClassNotFoundException 
	 * @throws InterruptedException 
	 * @throws URISyntaxException 
	 */
	protected void init(int m, int n, int d) throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException{
		MatrixInitialization obj = new MatrixInitialization();
		//Initialize U
		Configuration conf = FileOperators.getConfiguration();
		obj.init(d, 0, inputPath, conf.get("hadoop.cache.path")+"U/",true);
		obj.perform(conf);
		//Initialize uBias and vBias
		BiasGenerator.getBias(inputPath, m, n);
		List<String> line = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"vBias.dat");
		vBias = new double[line.size()];
		for(int i=0;i<line.size();i++)
			vBias[i] = Double.parseDouble(line.get(i));
	}
	
	/**
	 * Perform prediction
	 * @param inputPath, input path of click data
	 * @param outputPath, output path for prediction results
	 * @param numD, number of latent dimensions
	 * @throws IOException
	 * @throws URISyntaxException
	 * @throws InterruptedException
	 * @throws ClassNotFoundException
	 */
	public void predictPair(String inputPath, int numD) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numD", numD);
		Job job = new Job(conf);
		job.setJarByClass(HybridBiasMF.class);
		job.setJobName("DistHyBridBiasMF-Prediction");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(SGDMapper.class);
		job.setReducerClass(PredictionReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"uBias/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.data.path")+"Fu/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"/prediction-DistHyBridBiasMF/"));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"vBias.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.data.path")+"Fv.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"B.dat"), job.getConfiguration());
		job.waitForCompletion(true);
	}
	
	/**
	 * Reducer for prediction
	 * @author tigerzhong
	 *
	 */
	public static class PredictionReducer extends Reducer<LongWritable, MatrixRowWritable, Text, DoubleWritable>{
		/**
		 * Latent matrix for the current user
		 */
		protected double[] U = null;
		/**
		 * Latent matrix for items
		 */
		protected double[][] V = null;
		/**
		 * Current user bias
		 */
		protected double uBias = 0.0;
		/**
		 * Item bias
		 */
		protected double[] vBias = null;
		/**
		 * Feature vector for the current user
		 */
		protected double[] Fu = null;
		/**
		 * Feature matrix for items
		 */
		protected double[][] Fv = null;
		/**
		 * Regression matrix
		 */
		protected double[][] B = null;
		/**
		 * Number of latent dimension
		 */
		protected int numD = 10;
		
		protected double a = 0.0;
		
		protected double b = 0.0;
		
		protected double c = 0.0;
		/**
		 * Output key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		/**
		 * Loss function
		 */
		private HybridBiasLoss lossFunc = null;
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			List<Integer> idList = new ArrayList<Integer>();
			//Get Data
			for(MatrixRowWritable value : values){
				if(value.isSparse()){ //data
					idList.add((int)value.getFirstID());
				} else {			  //
					double[] vec = value.getVector();
					if(vec[0]<0.5){ 		//latent factors for user
						for(int i=0;i<vec.length-1;i++)
							U[i] = vec[i+1];
					} else if (vec[0]<1.5){ //bias factors for user
						uBias = vec[1];
					} else if (vec[0]<2) {  //features for user
						for(int i=0;i<vec.length-1;i++)
							Fu[i] = vec[i+1];
					}
				}
			}
			for(Integer q : idList){
				double r = lossFunc.getPrediction(U, V[q], B, Fu, Fv[q], uBias, vBias[q], a, b, c);
				outKey.set(key+","+q);
				outValue.set(r);
				context.write(outKey, outValue);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			//Read U, V, vBias
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			this.numD = conf.getInt("model.numD", 10);
			this.a = conf.getFloat("model.a", 1.0f);
			this.b = conf.getFloat("model.b", 1.0f);
			this.c = conf.getFloat("model.c", 1.0f);
			V = MatrixIO.readDense2DArrayFromText(paths[0].toString());
			vBias =  MatrixIO.readDense1DArrayFromText(paths[1].toString());
			Fv = MatrixIO.readDense2DArrayFromText(paths[2].toString());
			B = MatrixIO.readDense2DArrayFromText(paths[3].toString());
			lossFunc = new HybridBiasLoss();
		}
	}

	@Deprecated
	public void predictAll() throws Exception {
		
	}

}
