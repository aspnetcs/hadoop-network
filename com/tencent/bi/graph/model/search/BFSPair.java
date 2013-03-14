package com.tencent.bi.graph.model.search;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.LongPairWritable;

/**
 * Breath First Search
 * @author tigerzhong
 *
 */
public class BFSPair {
	/**
	 * Number of iterations
	 */
	protected int numIt = 100;
	/**
	 * Input path
	 */
	protected String inputPath = "";
	
	public void initModel(int numIt, String inputPath){
		this.numIt = numIt;
		this.inputPath = inputPath;
	}
	
	/**
	 * Perform search
	 * @throws Exception 
	 */
	public void performSearch() throws Exception{
		for(int it=0;it<numIt;it++){
			Configuration conf = FileOperators.getConfiguration();
			conf.set("pair.path", inputPath);
			//First MR, flooding
			Job job = new Job(conf);
			job.setJarByClass(BFSPair.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongPairWritable.class);
			job.setMapOutputValueClass(LongWritable.class);
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(NullWritable.class);
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
			conf.set("mapred.textoutputformat.separator", ",");
			job = new Job(conf);
			job.setJarByClass(BFSPair.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(NullWritable.class);
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(NullWritable.class);
			job.setMapperClass(Mapper.class);
			job.setCombinerClass(SelectionReducer.class);
			job.setReducerClass(SelectionReducer.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"BFSPair/"));	//Distance
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"BFSPair/"+(it+1)+"/"));
			job.waitForCompletion(true);
			//Postprocess
			FileSystem fs = FileSystem.get(conf); 
			fs.delete(new Path(conf.get("hadoop.tmp.path")+"BFSPair/"), true);
		}
	}
	
	public static class BFSMapper extends Mapper<LongWritable, Text, LongPairWritable, LongWritable>{
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private LongWritable outValue = new LongWritable();
		
		private String pairPath = "";
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			if(path.indexOf(pairPath)>-1){ //relation
				String[] pair = value.toString().split("\t");
				outKey.set(Long.parseLong(pair[1]), 1);
				outValue.set(Long.parseLong(pair[0]));
			} else { 			 		   //interaction
				outKey.set(Long.parseLong(value.toString()), 0);
				outValue.set(0);
			}
			context.write(outKey, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			pairPath = context.getConfiguration().get("pair.path");
		}
	}
	
	/**
	 * For flooding from queued nodes to others
	 * @author tigerzhong
	 *
	 */
	public static class BFSReducer extends Reducer<LongPairWritable, LongWritable, LongWritable, NullWritable>{
		/**
		 * Output key 
		 */
		private LongWritable outKey = new LongWritable();
			
		@Override
		public void reduce(LongPairWritable key, Iterable<LongWritable> values, Context context) throws IOException, InterruptedException {
			boolean hasValue = false;	//the current node is reachable?
			for(LongWritable value : values){
				if(key.getSecond()==0){ 	//list
					hasValue = true;
					outKey.set(key.getFirst());
					context.write(outKey, NullWritable.get());
				} else {					//pair
					if(hasValue){			//flooding
						long id = value.get();
						outKey.set(id);
						context.write(outKey, NullWritable.get());
					} else break;
				}
			}
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
	public static class SelectionReducer extends Reducer<LongWritable, NullWritable, LongWritable, NullWritable>{

		@Override
		public void reduce(LongWritable key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
			context.write(key, NullWritable.get());
		}

	}
}
