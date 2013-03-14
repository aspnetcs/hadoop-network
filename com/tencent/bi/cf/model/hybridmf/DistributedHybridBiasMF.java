package com.tencent.bi.cf.model.hybridmf;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

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
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.model.common.BiasGenerator;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.cf.optimization.gradient.hybrid.DistributedSGDHybridBias;
import com.tencent.bi.cf.optimization.gradient.hybrid.SquareBiasHybridLoss;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Mult;

/**
 * HybridMF with Bias
 * Distributed HybridMF which decomposes the click matrix and learns a regression matrix while using user and item bias
 * It explores the multiple inputs to handle huge user number
 * R = UV' + Fu*B*Fv + uBias + vBias
 * @author tigerzhong
 *
 */
public class DistributedHybridBiasMF extends AbstractHybridMF {
	/**
	 * Regression matrix
	 */
	protected DenseDoubleMatrix2D B = null;
	/**
	 * Feature matrix for items
	 */
	protected DenseDoubleMatrix2D Fv = null;
	/**
	 * Item bias
	 */
	protected DenseDoubleMatrix1D vBias = null;
	/**
	 * Optimization method
	 */
	protected DistributedSGDHybridBias solver = null;
	
	@Override
	public void initModel(int m, int n, int d, int fm, int fn, String solverName, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		super.initModel(-1, n, d, "", lambda, learningRate, numIt, inputPath);
		solver = (DistributedSGDHybridBias) Class.forName(solverName).newInstance();
		init(m, n, d);
		B = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(fm,fn);
		B.assign(Mult.div(fm));
	}
	
	
	@Override
	public void buildModel() throws Exception {
		SquareBiasHybridLoss lossFunc = new SquareBiasHybridLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.V.columns());
			solver.update(null, super.V, this.B, null, this.vBias, super.inputPath, conf.get("hadoop.output.path")+"/DistributedHybridBiasMF/"+i+"/");
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
		BiasGenerator.getBias(super.inputPath, m, n);
		List<String> line = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"vBias.dat");
		vBias = (DenseDoubleMatrix1D) DoubleFactory1D.dense.make(line.size());	//load item bias since it is small
		for(int i=0;i<line.size();i++) vBias.set(i, Double.parseDouble(line.get(i)));
	}
	
	@Override
	public void predictPair(String inputPath, int numD) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numD", numD);
		Job job = new Job(conf);
		job.setJarByClass(DistributedHybridBiasMF.class);
		job.setJobName("DistHyBridBiasMF-Prediction");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PredictionMapper.class);
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
	 * Mapper for prediction
	 * @author tigerzhong
	 *
	 */
	public static class PredictionMapper extends Mapper<LongWritable, Text, Text, Text>{
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String items[] = value.toString().split(",",2);
			String outLine = items[1];
			if(path.indexOf("/U")>-1){ 				//latent factors for user
				outLine += ",U";
			} else if (path.indexOf("/uBias")>-1){ 	//bias factors for user
				outLine += ",u";
			} else if (path.indexOf("/Fu")>-1) {	//user features
				outLine += ",fu";
			}
			outKey.set(items[0]);
			outText.set(outLine);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * Reducer for prediction
	 * @author tigerzhong
	 *
	 */
	public static class PredictionReducer extends Reducer<Text, Text, Text, DoubleWritable>{
		/**
		 * Latent matrix for the current user
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Latent matrix for items
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Current user bias
		 */
		protected double uBias = 0.0;
		/**
		 * Item bias
		 */
		protected DenseDoubleMatrix1D vBias = null;
		/**
		 * Feature vector for the current user
		 */
		protected DenseDoubleMatrix1D Fu = null;
		/**
		 * Feature matrix for items
		 */
		protected DenseDoubleMatrix2D Fv = null;
		/**
		 * Regression matrix
		 */
		protected DenseDoubleMatrix2D B = null;
		/**
		 * Number of latent dimension
		 */
		protected int numD = 10;
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
		private SquareBiasHybridLoss lossFunc = null;
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Iterator<Text> it = values.iterator();
			List<Integer> idList = new ArrayList<Integer>();
			//Get Data
			while(it.hasNext()){
				String inLine = it.next().toString().trim();
				String[] items = inLine.split(",");
				if(items[items.length-1].equals("u")){			//uBias
					this.uBias = Double.parseDouble(items[0]);
				} else if (items[items.length-1].equals("U")){	//U
					U = new DenseDoubleMatrix1D(this.numD);
					for(int i=0;i<items.length-1;i++)
						this.U.set(i, Double.parseDouble(items[i]));		
				} else if(items[items.length-1].equals("fu")){	//Fu
					Fu = new DenseDoubleMatrix1D(items.length-1);
					for(int i=0;i<items.length-1;i++)
						this.Fu.set(i, Double.parseDouble(items[i]));	
				} else{ 										//data
					idList.add(Integer.parseInt(items[0]));
				}
			}

			for(Integer q : idList){
				double r = lossFunc.getPrediction(U, V.viewRow(q), B, Fu, Fv.viewRow(q), uBias, vBias.get(q));
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
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			vBias = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[1].toString()).viewColumn(0);
			Fv = MatrixIO.readDenseMatrixFromText(paths[2].toString());
			B = MatrixIO.readDenseMatrixFromText(paths[3].toString());
			lossFunc = new SquareBiasHybridLoss();
		}
	}
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public double predict(int p, int q, int o) {
		//Ignore
		return 0;
	}


	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) {
		return 0;
	}

	@Deprecated
	public void predictAll() throws Exception {
		
	}

}
