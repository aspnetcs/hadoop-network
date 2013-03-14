package com.tencent.bi.cf.optimization.als;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
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
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.cf.optimization.Loss;
import com.tencent.bi.cf.optimization.MFSolver;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;
import com.tencent.bi.utils.serialization.PairRowWritable;

//import cern.colt.list.DoubleArrayList;
//import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
//import cern.colt.matrix.doublealgo.Transform;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
//import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.linalg.Algebra;
import cern.jet.math.PlusMult;

/**
 * Alternative Least Square for Matrix Factorization with Square Loss
 * U = RV(V'V+\lambda*I)^{-1}
 * V = R'U(U'U+\lambda*I)^{-1}
 * @author tigerzhong
 *
 */

public class GaussianSolver implements MFSolver{
	/**
	 * Number of latent factors
	 */
	protected static int numD = 10;
	/**
	 * Regularization parameter
	 */
	protected static double lambda = 0.0;
	/**
	 * Number of users
	 */
	protected static int numUser;
	/**
	 * Number of items	
	 */
	protected static int numItem;
	
	@Override
	public void initialize(Loss lossFunc, double lambda, double learningRate,
			int numD) throws Exception {
		GaussianSolver.numD = numD;
		GaussianSolver.lambda = lambda;
	}

	@Override
	public void update(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath) throws Exception {
		numUser = (int) U.get(0, 0);
		numItem = (int) V.get(0, 0);
		performUpdate(new DenseDoubleMatrix2D(1,1), null, inputPath, outputPath);	//update V
		performUpdate(null, new DenseDoubleMatrix2D(1,1), inputPath, outputPath);	//update U
	}
	
	/**
	 * Update latent matrix
	 * @param U, latent matrix for users
	 * @param V, latent matrix for items
	 * @param inputPath, data input path
	 * @param outputPath, data output path
	 * @throws Exception
	 */
	public void performUpdate(DenseDoubleMatrix2D U, DenseDoubleMatrix2D V, String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		conf.setInt("model.numD", numD);
		conf.setFloat("model.lambda", (float) lambda);
		conf.setInt("model.numU", numUser);
		conf.setInt("model.numV", numItem);
		//First MR, transforming dataset
		Job job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-ALS-Phase1");
		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(PairRowWritable.class);
		job.setMapperClass(MatchMapper.class);
		job.setReducerClass(MatchReducer.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		FileInputFormat.addInputPath(job, new Path(inputPath));
		if(U==null){		//update U
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"V/"));
			job.getConfiguration().setBoolean("model.u", true);
		} else {			//update V
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
			job.getConfiguration().setBoolean("model.u", false);
		}
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"ALS/"));
		job.waitForCompletion(true);
		
		//Postprocess
		FileSystem fs = FileSystem.get(conf); 
		if(U==null)	
			fs.delete(new Path(conf.get("hadoop.cache.path")+"U/"), true);
		else fs.delete(new Path(conf.get("hadoop.cache.path")+"V/"), true);
		
		//Second MR, computing each v_j*r_ij and then sum_j(v_j*r_ij)
		conf.set("mapred.textoutputformat.separator", ",");
		job = new Job(conf);
		job.setJarByClass(GaussianSolver.class);
		job.setJobName("MF-ALS-Phase2");	
		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(PairRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(Text.class);
		job.setMapperClass(UpdateMapper.class);
		job.setReducerClass(UpdateReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"ALS/"));
		if(U==null)
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"U/"));
		else FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"V/"));
		job.waitForCompletion(true);
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"ALS/"), true);	
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	 * First mapper, matching the row and distributing the data
	 */
	public static class MatchMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable>{
		/**
		 * Output Key
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		/**
		 * Update U or V
		 */
		private boolean isU = false;
		
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String[] items = value.toString().trim().split(",");
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			if(path.indexOf("/V")>-1 || path.indexOf("/U")>-1){	//vector
				outKey.set(Long.parseLong(items[0]));
				DenseDoubleMatrix1D vec = new DenseDoubleMatrix1D(items.length-1);
				for(int i=0;i<items.length-1;i++)
					vec.setQuick(i, Double.parseDouble(items[i+1]));
				outValue.set(vec);
				context.getCounter("Eval", "Cnt").increment(1);
			} else {//data
				double r = Double.parseDouble(items[2]);
				if(!isU) { // user id as key, update V
					outKey.set(Long.parseLong(items[0]));
					long y = Long.parseLong(items[1]);
					outValue.set(y, r);			
				} else { // item id as key, update U
					outKey.set(Long.parseLong(items[1]));
					long y = Long.parseLong(items[0]);
					outValue.set(y, r);
				}
			}
			context.write(outKey, outValue);
		}
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			isU = conf.getBoolean("model.u", false);
		}
	}
	/**
	 * First reducer, expanding the data to [i, r_ij, u_i] 
	 * @author tigerzhong
	 *
	 */
	public static class MatchReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, PairRowWritable> {

		private PairRowWritable outValue = new PairRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			//SparseDoubleMatrix1D vec = new SparseDoubleMatrix1D(Math.max(numUser, numItem));
			List<Long> ids = new ArrayList<Long>();
			List<Double> vals = new ArrayList<Double>();
			Iterator<MatrixRowWritable> it = values.iterator();
			while(it.hasNext()){
				MatrixRowWritable currentItem = it.next();
				if(currentItem.isSparse()){
					ids.add(currentItem.getFirstID());
					vals.add(currentItem.getFirstVal());
					//vec.setQuick(currentItem.getFirstID(), currentItem.getFirstVal());
				}
				else outValue.setFactors(currentItem.getDenseVector());					
			}
			outValue.setRatings(ids, vals, Math.max(numUser, numItem));
			//outValue.setRatings(vec);
			//Output
			context.write(key, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			numUser = conf.getInt("model.numU", 0);
			numItem = conf.getInt("model.numV", 0);
		}
	}
	
///////////////////////////////////////////////////////////////////////////////////////////////
	
	/**
	 * Second mapper, distributing data
	 */
	public static class UpdateMapper extends Mapper<LongWritable, PairRowWritable, LongWritable, PairRowWritable>{
		/**
		 * Output Key
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private PairRowWritable outValue = new PairRowWritable();
		
		@Override
		public void map(LongWritable key, PairRowWritable value, Context context)
				throws IOException, InterruptedException {
			//SparseDoubleMatrix1D ratings = value.getRatingVector();
	  		//IntArrayList Y = new IntArrayList();
	  		//DoubleArrayList R = new DoubleArrayList();
	  		//ratings.getNonZeros(Y, R);
	    	//int nnz = R.size();
			long[] ids = value.getRowIDs();
			double[] vals = value.getRatings();
			int nnz = ids.length;
			for(int i=0;i<nnz;i++){
				//outValue.setRating(Y.getQuick(i), R.getQuick(i));
				//outValue.setFactors(value.getFactors());
				//outKey.set(Y.getQuick(i));
				outValue.setRating(ids[i], vals[i]);
				outValue.setFectors(value.getFactorArray());
				outKey.set(ids[i]);
				context.write(outKey, outValue);
			}
		}
	}
	
	/**
	 * Second reducer, updating latent vectors
	 * @author tigerzhong
	 *
	 */
	public static class UpdateReducer extends Reducer<LongWritable, PairRowWritable, LongWritable, Text> {
		/**
		 * Identity matrix
		 */
		protected DoubleMatrix2D Ek = null;
		/**
		 * Matrix for V'V
		 */
		protected DoubleMatrix2D A = null;
		/**
		 * Matrix for RV
		 */
		protected DoubleMatrix1D B = null;
		/**
		 * Output value
		 */
		private Text outValue = new Text();
		@Override
		public void reduce(LongWritable key, Iterable<PairRowWritable> values, Context context) throws IOException, InterruptedException {
			DoubleMatrix2D Ai = new DenseDoubleMatrix2D(numD, numD);
			Algebra algebraObj = new Algebra();
			Iterator<PairRowWritable> it = values.iterator();
			while(it.hasNext()){			//Combining
				PairRowWritable line = it.next();
				DoubleMatrix1D V = line.getFactors();
				double r = line.getRatings()[0];
				algebraObj.multOuter(V, V, Ai);		
				A.assign(Ai, PlusMult.plusMult(1.0)); //A = A + V'V
				B.assign(V, PlusMult.plusMult(r));	//b = b + V'R	
				//A = Transform.plusMult(A, Ai, 1);//A = A + V'V
				//B = Transform.plusMult(B, V, r); //b = b + V'R	
			}
			DoubleMatrix2D invA = algebraObj.inverse(A);	  // A^-1 = inv(A)
			DoubleMatrix1D U = new DenseDoubleMatrix1D(numD); //Current vector
			U = invA.zMult(B, U);// U = A^-1 x b
			//Output
			outValue.set(StringUtils.array2String(U.toArray()));
			context.write(key, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			Configuration conf = context.getConfiguration();
			GaussianSolver.numD = conf.getInt("model.numD", 10);
			GaussianSolver.lambda = conf.getFloat("model.lambda", 0.005f);
			Ek = DoubleFactory2D.sparse.identity(numD);
			A = new DenseDoubleMatrix2D(numD, numD);
			B = new DenseDoubleMatrix1D(numD);
			A.assign(Ek, PlusMult.plusMult(lambda)); //A = A + lambda*Ek
			//A = Transform.plusMult(A, Ek, lambda); //A = A + lambda*Ek
		}
	}

}
