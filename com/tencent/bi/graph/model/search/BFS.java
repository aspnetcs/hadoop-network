package com.tencent.bi.graph.model.search;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Breath First Search
 * @author tigerzhong
 *
 */
public abstract class BFS {
	/**
	 * Number of nodes
	 */
	protected int numNode = 0;
	/**
	 * Number of iterations
	 */
	protected int numIt = 100;
	/**
	 * Input path
	 */
	protected String inputPath = "";
	/**
	 * Output path
	 */
	protected String outputPath = "";
	/**
	 * Visit node
	 * @param nodeID
	 * @param dist, distance value
	 */
	protected abstract void visit(long nodeID, double dist) throws Exception;
	/**
	 * Stop searching
	 * @return
	 */
	protected abstract boolean isFinished() throws Exception;
	
	public void initModel(int numNode, int numIt, String inputPath, String outputPath){
		this.numNode = numNode;
		this.numIt = numIt;
		this.inputPath = inputPath;
		this.outputPath = outputPath;
	}
	
	/**
	 * Perform search
	 * @throws Exception 
	 */
	protected void performSearch() throws Exception{
		for(int it=0;it<numIt;it++){
			Configuration conf = FileOperators.getConfiguration();
			//First MR, flooding
			Job job = new Job(conf);
			job.setJarByClass(BFS.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(MatrixRowWritable.class);
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(MatrixRowWritable.class);
			job.setMapperClass(Mapper.class);
			job.setReducerClass(BFSReducer.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(inputPath));
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"DistBFS/"+it+"/"));	//Distance
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"DistBFS/"+(it+1)+"/"));
			job.waitForCompletion(true);
			if(isFinished()) break;
		}
	}
	
	/**
	 * For flooding from queued nodes to others
	 * @author tigerzhong
	 *
	 */
	public static class BFSReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		/**
		 * Output key 
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
			
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			boolean hasValue = false;	//the current node is reachable?
			double dist = 0;			//the distance from the start node to the current node
			long[] adList = null;		//the adjacency node list of the current node
			double[] distList = null;	//the corresponding distance list 
			for(MatrixRowWritable value : values){
				if(!value.isSparse()){ 	//Distance
					hasValue = true;
					dist = value.getFirstVal();
				} else {				//ADList
					adList = value.getIDs(); 
					distList = value.getVector();
				}
			}
			if(hasValue){	//flooding
				for(int i=0;i<adList.length;i++){
					long id = adList[i];
					double cd = distList[i];
					outKey.set(id);
					outValue.set(dist+cd);
					context.write(outKey, outValue);
				}
			}
		}
	}
	
	/**
	 * Select the shortest distance
	 * @author tigerzhong
	 *
	 */
	public static class SelectionReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		/**
		 * Current searching object
		 */
		private BFS searchObj = null;
		/**
		 * Output value
		 */
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			double dist = Double.MAX_VALUE;
			for(MatrixRowWritable value : values){		//get the shortest distance
				double currentDist = value.getFirstVal();
				dist = Math.min(dist, currentDist);
			}
			try {
				searchObj.visit(key.get(), dist);
			} catch (Exception e) {
				e.printStackTrace();
			}
			outValue.set(dist);
			context.write(key, outValue);
		}
		
		@Override
		public void setup(Context context){
			try {
				searchObj = (BFS) Class.forName(context.getConfiguration().get("model.name")).newInstance();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}
}
