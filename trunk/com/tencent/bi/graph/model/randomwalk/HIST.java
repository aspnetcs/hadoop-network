package com.tencent.bi.graph.model.randomwalk;

import java.io.IOException;
import java.net.URISyntaxException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.jet.math.PlusMult;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

@Deprecated
public class HIST extends AbstractRWModel {
	
	protected void performRandomWalk(int iteration) throws Exception{
		update(iteration, false);
		update(iteration, true);
	}

	protected void update(int iteration, boolean isHub) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setFloat("model.alpha", (float) alpha);
		conf.setInt("model.numValues", numValues);
		conf.setInt("model.numNode", numNode);
		String cacheout = conf.get("hadoop.cache.path");
		String tmpout = conf.get("hadoop.tmp.path");
		//First MR, distributing data
		Job job = new Job(conf);
		job.setJarByClass(PageRank.class);
		job.setJobName("HIST-Dist-It-"+iteration);	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(RWDisADListReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.output.path")+"HIST-Dist-It-"+iteration+"/"));
		if(!isHub)
			FileInputFormat.addInputPath(job, new Path(cacheout+"H/"));
		else FileInputFormat.addInputPath(job, new Path(cacheout+"A/"));
		FileOutputFormat.setOutputPath(job, new Path(tmpout+"HIST-Dist/"));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		if(!isHub) fs.delete(new Path(cacheout+"H/"), true);
		else fs.delete(new Path(cacheout+"A/"), true);
		//Second MR, summarizing data
		job = new Job(conf);
		job.setJarByClass(PageRank.class);
		job.setJobName("PageRank-Sum-It-"+iteration);	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setMapperClass(Mapper.class);
		job.setReducerClass(RWSumReducer.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.addInputPath(job, new Path(tmpout+"HIST-Dist/"));
		if(!isHub) FileOutputFormat.setOutputPath(job, new Path(cacheout+"A/"));
		else FileOutputFormat.setOutputPath(job, new Path(cacheout+"H/"));
		job.waitForCompletion(true);
		fs.delete(new Path(tmpout+"HIST-Dist/"), true);
	}
	
	/**
	 * Reducer for distributing data
	 * @author tigerzhong
	 *
	 */
	public static class RWDisADListReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			double[] valueList = null;
			double[] probList = null;
			long[] idList = null;
			for(MatrixRowWritable item : values){
				if(item.isSparse()){
					probList = item.getVector();
					idList = item.getIDs();
				}
				else valueList = item.getVector();
			}
	  		int numN = idList.length;
	  		for(int i=0;i<numN;i++){
	  			double[] probValueList = new double[valueList.length];
	  			for(int j=0;j<valueList.length;j++)
	  				probValueList[j] = valueList[j]*probList[i]/numN;
	  			outKey.set(idList[i]);
	  			outValue.set(probValueList);
	  			context.write(outKey, outValue);
	  		}
		}		
	}
	
	/**
	 * Reducer for summarizing data
	 * @author tigerzhong
	 *
	 */
	public static class RWSumReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{

		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			DenseDoubleMatrix1D valueList = new DenseDoubleMatrix1D(numValues);
			for(MatrixRowWritable item : values)
				valueList.assign(item.getDenseVector(), PlusMult.plusMult(alpha));
			outValue.set(valueList);
			context.write(key, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			alpha = context.getConfiguration().getFloat("model.alpha", (float) 0.9);
			numValues = context.getConfiguration().getInt("model.numValues", 1);
		}
	}
}
