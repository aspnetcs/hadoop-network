package com.tencent.bi.cf.model.biasmf;

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

import com.tencent.bi.cf.model.AbstractMF;
import com.tencent.bi.cf.model.common.BiasGenerator;
import com.tencent.bi.cf.model.common.MatrixInitialization;
import com.tencent.bi.cf.optimization.gradient.bias.SquareBiasLoss;
import com.tencent.bi.cf.optimization.gradient.bias.DistributedSGDBias;
import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleFactory1D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Distributed matrix factorization with bias for huge user number
 * R = UV' + uBias + vBias
 * @author tigerzhong
 *
 */
public class DistributedBiasMF extends AbstractMF {
	/**
	 * User bias
	 */
	protected DenseDoubleMatrix1D uBias = null;
	/**
	 * Item bias
	 */
	protected DenseDoubleMatrix1D vBias = null;
	/**
	 * Optimization method
	 */
	protected DistributedSGDBias solver = null;
	
	@Override
	public void initModel(int m, int n, int d, String solverName, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		super.initModel(-1, n, d, "", lambda, learningRate, numIt, inputPath);
		solver = (DistributedSGDBias) Class.forName(solverName).newInstance();
		init(m, n, d);
	}
	
	@Override
	public void buildModel() throws Exception {
		SquareBiasLoss lossFunc = new SquareBiasLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, super.V.columns());
			solver.update(null, super.V, null, this.vBias, super.inputPath, conf.get("hadoop.output.path")+"/DistBiasMF/"+i+"/");
		}	
	}
	
	/**
	 * Initialize matrix
	 * @param m
	 * @param d
	 * @throws IOException
	 * @throws ClassNotFoundException 
	 * @throws InterruptedException 
	 * @throws URISyntaxException 
	 */
	private void init(int m, int n, int d) throws IOException, InterruptedException, ClassNotFoundException, URISyntaxException{
		Configuration conf = FileOperators.getConfiguration();
		MatrixInitialization obj = new MatrixInitialization();
		//Initialize U
		obj.init(d, 0, this.inputPath, conf.get("hadoop.cache.path")+"U/",true);
		obj.perform(conf);
		//Initialize uBias and vBias
		BiasGenerator.getBias(super.inputPath ,m,n);
		List<String> line = DataOperators.readTextFromHDFS(new Configuration(), conf.get("hadoop.cache.path")+"vBias.dat");
		vBias = (DenseDoubleMatrix1D) DoubleFactory1D.dense.make(line.size());
		for(int i=0;i<line.size();i++){
			vBias.set(i, Double.parseDouble(line.get(i)));
		}
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
	@Override
	public void predictPair(String inputPath, int numD) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numD", numD);
		Job job = new Job(conf);
		job.setJarByClass(AbstractMF.class);
		job.setJobName("DistBiasMF-Prediction");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PredictionMapper.class);
		job.setReducerClass(PredictionReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"uBias/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"prediction-DistributedBiasMF/"));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"vBias.dat"), job.getConfiguration());
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
			}
			outKey.set(items[0]);
			outText.set(outLine);
			context.write(outKey, outText);
		}
	}
	
	/**
	 * Mapper for prediction
	 * @author tigerzhong
	 *
	 */
	public static class PredictionReducer extends Reducer<Text, Text, Text, DoubleWritable>{
		/**
		 * Latent Matrix for the Current User
		 */
		protected DenseDoubleMatrix1D U = null;
		/**
		 * Latent Matrix for Items
		 */
		protected DenseDoubleMatrix2D V = null;
		/**
		 * Current User Bias
		 */
		protected double uBias = 0.0;
		/**
		 * Item Bias
		 */
		protected DenseDoubleMatrix1D vBias = null;
		/**
		 * Number of latent dimension
		 */
		protected int numD = 10;
		/**
		 * Output Key
		 */
		private Text outKey = new Text();
		/**
		 * Output value
		 */
		private DoubleWritable outValue = new DoubleWritable();
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Iterator<Text> it = values.iterator();
			List<Integer> idList = new ArrayList<Integer>();
			while(it.hasNext()){
				//Get Data
				String inLine = it.next().toString().trim();
				String[] items = inLine.split(",");
				if(items[items.length-1].equals("u")){			//uBias
					uBias = Double.parseDouble(items[0]);
				} else if (items[items.length-1].equals("U")){	//U
					for(int i=0;i<items.length-1;i++)
						U.set(i, Double.parseDouble(items[i]));		
				} else{ 										//data
					idList.add(Integer.parseInt(items[0]));
				}
			}
			for(Integer q : idList){
				double r = U.zDotProduct(V.viewRow(q))+ uBias + vBias.get(q);
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
			U = new DenseDoubleMatrix1D(this.numD);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			vBias = (DenseDoubleMatrix1D) MatrixIO.readDenseMatrixFromText(paths[1].toString()).viewColumn(0);
		}
	}

	@Override
	public void predictAll() throws Exception { 
		Configuration conf = FileOperators.getConfiguration();
		conf.set("mapred.textoutputformat.separator", ",");
		Job job = new Job(conf);
		job.setJarByClass(DistributedBiasMF.class);
		job.setJobName("MFBias-Prediction-All");
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PredictionMapper.class);
		job.setReducerClass(MultiplyReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"uBias/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"MF-Prediction/"));
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"V.dat"), job.getConfiguration());
		DistributedCache.addCacheFile(new URI(conf.get("hadoop.cache.path")+"vBias.dat"), job.getConfiguration());
		job.waitForCompletion(true);
	}
	
	@Override
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) {
		return u.zDotProduct(v)+s.get(0)+s.get(1);
	}
	
	/**
	 * Mapper, distributing data
	 */
	public static class MultiplyReducer extends Reducer<Text, Text, Text, Text>{
		/**
		 * Output value
		 */
		private Text outText = new Text();
		
		private DoubleMatrix2D V = null;
		
		private DoubleMatrix1D vBias = null;
		
		private DistributedBiasMF model = null;
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			DoubleMatrix1D u = null;
			double uBias = 0.0;
			for(Text value : values){
				String[] items = value.toString().split(",");
				if(items[items.length-1].equals("u")){
					uBias = Double.parseDouble(items[0]);
				} else if(items[items.length-1].equals("U")){
					u = new DenseDoubleMatrix1D(items.length-1);
					for(int i=0;i<items.length-1;i++){
						u.set(i, Double.parseDouble(items[i]));
					}
				}
			}
			StringBuilder outLine = new StringBuilder();
			for(int i=0;i<V.rows();i++){
				if(i!=0) outLine.append(",");
				DoubleMatrix1D s = new DenseDoubleMatrix1D(2);
				s.setQuick(0, uBias); s.setQuick(1, vBias.get(i));
				outLine.append(model.predict(u, V.viewRow(i), s));
			}
			outText.set(outLine.toString());
			context.write(key, outText);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			model = new DistributedBiasMF();
			Path[] paths = DistributedCache.getLocalCacheFiles(conf);
			V = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			vBias = MatrixIO.readDenseMatrixFromText(paths[1].toString()).viewColumn(0);
		}
	}
	
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	@Deprecated
	public double predict(int p, int q, int o) { return 0;	}

}
