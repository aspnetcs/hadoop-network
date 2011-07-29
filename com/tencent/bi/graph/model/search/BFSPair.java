package com.tencent.bi.graph.model.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.LongPairWritable;

/**
 * Breath First Search
 * @author tigerzhong
 *
 */
public class BFSPair {
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
	
	protected long targetID = 0;
	
	protected long startID = 0;
	
	public void initModel(int numNode, int numIt, long na, long nb, String inputPath, String outputPath){
		this.numNode = numNode;
		this.numIt = numIt;
		this.inputPath = inputPath;
		this.outputPath = outputPath;
		this.targetID = nb;
		this.startID = na;
	}
	
	/**
	 * Perform search
	 * @throws Exception 
	 */
	public void performSearch() throws Exception{
		for(int it=0;it<numIt;it++){
			Configuration conf = FileOperators.getConfiguration();
			//First MR, flooding
			Job job = new Job(conf);
			job.setJarByClass(BFSPair.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongPairWritable.class);
			job.setMapOutputValueClass(LongWritable.class);
			job.setOutputKeyClass(LongPairWritable.class);
			job.setOutputValueClass(LongWritable.class);
			job.setPartitionerClass(FirstPartitioner.class);
			job.setSortComparatorClass(KeyComparator.class);
			job.setGroupingComparatorClass(GroupComparator.class);
			job.setMapperClass(BFSMapper.class);
			job.setReducerClass(BFSReducer.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(inputPath));
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"BFSPair/"+it+"/"));	//Distance
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"BFSPair/"));
			job.waitForCompletion(true);
			//Second MR, picking the max value
			conf.setLong("model.target", targetID);
			job = new Job(conf);
			job.setJarByClass(BFSPair.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongPairWritable.class);
			job.setMapOutputValueClass(LongWritable.class);
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(LongWritable.class);
			job.setMapperClass(Mapper.class);
			job.setReducerClass(SelectionReducer.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(inputPath));
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"BFSPair/"));	//Distance
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"BFSPair/"+(it+1)+"/"));
			job.waitForCompletion(true);
			//Postprocess
			FileSystem fs = FileSystem.get(conf); 
			fs.delete(new Path(conf.get("hadoop.tmp.path")+"BFSPair/"), true);
			if(fs.exists(new Path(conf.get("hadoop.tmp.path")+"Reachable.finished"))) break;
		}
	}
	
	public static class BFSMapper extends Mapper<LongWritable, Text, LongPairWritable, LongWritable>{
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split(",");
			if(items.length==1){ //relation
				String[] pair = items[0].split("\t");
				outKey.set(Long.parseLong(pair[0]), 1);
				outValue.set(Long.parseLong(pair[1]));
			} else { 			 //interaction
				outKey.set(Long.parseLong(items[0]), 0);
				outValue.set(Long.parseLong(items[1]));
			}
			context.write(outKey, outValue);
		}
	}
	
	/**
	 * For flooding from queued nodes to others
	 * @author tigerzhong
	 *
	 */
	public static class BFSReducer extends Reducer<LongPairWritable, LongWritable, LongWritable, LongWritable>{
		/**
		 * Output key 
		 */
		private LongWritable outKey = new LongWritable();
		/**
		 * Output value
		 */
		private LongWritable outValue = new LongWritable();
			
		@Override
		public void reduce(LongPairWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			boolean hasValue = false;	//the current node is reachable?
			long dist = 0;				//the distance from the start node to the current node
			for(LongWritable value : values){
				if(key.getSecond()==0){ 	//distance
					hasValue = true;
					dist = value.get();
				} else {					//pair
					if(hasValue){	//flooding
						long id = value.get();
						outKey.set(id);
						outValue.set(dist+1);
						context.write(outKey, outValue);
					}
				}
			}
			outKey.set(key.getFirst());
			outValue.set(dist);
			context.write(outKey, outValue);
		}
	}
	
	public static class FirstPartitioner extends Partitioner<LongPairWritable, LongWritable> {
		
		HashPartitioner<Long, LongWritable> p = new HashPartitioner<Long, LongWritable>();
		@Override
		public int getPartition(LongPairWritable key, LongWritable value, int numPartitions) {
			return p.getPartition(key.getFirst(), value, numPartitions);
		}
		
	}
	
	public static class KeyComparator extends WritableComparator {
	    
		protected KeyComparator() {
	    	super(LongPairWritable.class, true);
	    }
	    
	    @SuppressWarnings("rawtypes")
		@Override
	    public int compare(WritableComparable w1, WritableComparable w2) {
	    	LongPairWritable ip1 = (LongPairWritable) w1;
	    	LongPairWritable ip2 = (LongPairWritable) w2;
			return ip1.compareTo(ip2);
	    }
	}
	
	 public static class GroupComparator extends WritableComparator {
		 
		 protected GroupComparator() {
			 super(LongPairWritable.class, true);
		 }
		 
		 @SuppressWarnings("rawtypes")
		 @Override
		 public int compare(WritableComparable w1, WritableComparable w2) {
			 LongPairWritable ip1 = (LongPairWritable) w1;
			 LongPairWritable ip2 = (LongPairWritable) w2;
			 if(ip1.getFirst()==ip2.getFirst()) return 0;
			 else return ip1.getFirst()>ip2.getFirst() ? 1 : -1;
		 }
	 }	
	
	/**
	 * Select the shortest distance
	 * @author tigerzhong
	 *
	 */
	public static class SelectionReducer extends Reducer<LongWritable, LongWritable, LongWritable, LongWritable>{
		/**
		 * Output value
		 */
		private LongWritable outValue = new LongWritable();
		
		private long nb = 0;
		
		@Override
		public void reduce(LongWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			long dist = Long.MAX_VALUE;
			for(LongWritable value : values){		//get the shortest distance
				long currentDist = value.get();
				dist = Math.min(dist, currentDist);
			}
			outValue.set(dist);
			context.write(key, outValue);
			if(nb == key.get()){
				List<String> res = new ArrayList<String>();
				res.add(nb+","+dist);
				DataOperators.saveTextToHDFS(context.getConfiguration(), context.getConfiguration().get("hadoop.tmp.path")+"Reachable.finished", res, true);
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			nb = context.getConfiguration().getLong("model.target",0);
		}
	}
}
