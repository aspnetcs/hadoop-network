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

import com.tencent.bi.cf.optimization.gradient.hybrid.DistributedSGDRegression;
import com.tencent.bi.cf.optimization.gradient.hybrid.SquareRegressionLoss;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.io.MatrixIO;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Mult;

/**
 * Regression MF
 * Distributed Regression MF which decomposes the click matrix and learns a regression matrix
 * It explores the multiple inputs to handle huge user number
 * R = Fu*B*Fv
 * @author tigerzhong
 *
 */
public class DistributedRegressionMF extends AbstractHybridMF {
	/**
	 * Regression matrix
	 */
	protected DenseDoubleMatrix2D B = null;
	/**
	 * Feature matrix for items
	 */
	protected DenseDoubleMatrix2D Fv = null;
	/**
	 * Optimization method
	 */
	protected DistributedSGDRegression solver = null;
	
	@Override
	public void initModel(int m, int n, int d, int fm, int fn, String solverName, double lambda, double learningRate, int numIt, String inputPath) throws Exception{
		super.initModel(-1, -1, 0, "", lambda, learningRate, numIt, inputPath);
		solver = (DistributedSGDRegression) Class.forName(solverName).newInstance();
		B = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(fm,fn);
		B.assign(Mult.div(fm));
	}
	
	
	@Override
	public void buildModel() throws Exception {
		SquareRegressionLoss lossFunc = new SquareRegressionLoss();
		Configuration conf = FileOperators.getConfiguration();
		for(int i=0;i<super.numIt;i++){
			solver.initialize(lossFunc, super.lambda, super.learningRate, 0);
			solver.update(this.B, super.inputPath, conf.get("hadoop.output.path")+"/DistRegressionMF-"+learningRate+"/"+i+"/");
		}			
	}
	
	@Override
	public void predictPair(String inputPath, int numD) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(DistributedRegressionMF.class);
		job.setJobName("DistRegressionMF-Prediction");		
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(PredictionMapper.class);
		job.setReducerClass(PredictionReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.data.path")+"Fu/"));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.output.path")+"/prediction-DistRegressionMF/"));
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
			if (path.indexOf("/Fu")>-1) {	//user features
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
		private SquareRegressionLoss lossFunc = null;
		
		@Override
		public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			Iterator<Text> it = values.iterator();
			List<Integer> idList = new ArrayList<Integer>();
			//Get Data
			while(it.hasNext()){
				String inLine = it.next().toString().trim();
				String[] items = inLine.split(",");
				if(items[items.length-1].equals("fu")){	//Fu
					Fu = new DenseDoubleMatrix1D(items.length-1);
					for(int i=0;i<items.length-1;i++)
						this.Fu.set(i, Double.parseDouble(items[i]));	
				} else{ 										//data
					idList.add(Integer.parseInt(items[0]));
				}
			}

			for(Integer q : idList){
				double r = lossFunc.getPrediction(B, Fu, Fv.viewRow(q));
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
			Fv = MatrixIO.readDenseMatrixFromText(paths[0].toString());
			B = MatrixIO.readDenseMatrixFromText(paths[1].toString());
			lossFunc = new SquareRegressionLoss();
		}
	}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	@Deprecated
	public double predict(int p, int q, int o) { return 0; }


	@Deprecated
	public double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s) { return 0; }


	@Deprecated
	public void predictAll() throws Exception {	}	
}
