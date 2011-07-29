package com.tencent.bi.graph.model.randomwalk;

import java.io.IOException;
import java.net.URISyntaxException;

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
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Page Rank or Random Walk with Restart
 * pa(i+1) = \alpha*sum_{b\in N(a)}[pb(i)/N(b)] + (1-\alpha)*pa(0)
 * @author tigerzhong
 *
 */
public class PageRankPair extends AbstractRWModel{
	
	/**
	 * Random walk with restart
	 */
	protected void performRandomWalk(int iteration) throws IOException, URISyntaxException, InterruptedException, ClassNotFoundException{
		Configuration conf = FileOperators.getConfiguration();
		conf.setFloat("model.alpha", (float) alpha);
		conf.setInt("model.numValues", numValues);
		conf.setInt("model.numNode", numNode);
		//First MR, distributing data
		Job job = new Job(conf);
		job.setJarByClass(PageRankPair.class);
		job.setJobName("PageRank-Dist-It-"+iteration);	
		
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		
		job.setMapperClass(RWDisPairMapper.class);
		job.setPartitionerClass(FirstPartitioner.class);
		job.setSortComparatorClass(KeyComparator.class);
		job.setGroupingComparatorClass(GroupComparator.class);
		job.setReducerClass(RWDisPairReducer.class);
		
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"NumLinks/"));
		FileInputFormat.addInputPath(job, new Path(adListPath));
		FileInputFormat.addInputPath(job, new Path(inputPath));	//Importance value vectors
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(inputPath), true);
		
		//Second MR, summarizing data
		job = new Job(conf);
		job.setJarByClass(PageRankPair.class);
		job.setJobName("PageRank-Sum-It-"+iteration);	
		
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		
		job.setMapperClass(Mapper.class);
		job.setCombinerClass(RWSumReducer.class);
		job.setReducerClass(RWSumReducer.class);

		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"));
		FileOutputFormat.setOutputPath(job, new Path(inputPath));
		job.waitForCompletion(true);
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"), true);
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
			double[] valueList = new double[numValues];
			for(MatrixRowWritable item : values){
				double[] tmpList = item.viewVector();
				for(int i=0;i<numValues;i++)
					valueList[i]+=tmpList[i];
			}
			outValue.set(valueList);
			context.write(key, outValue);
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			numValues = context.getConfiguration().getInt("model.numValues", 1);
		}
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class RWDisPairMapper extends Mapper<LongWritable, Text, LongPairWritable, MatrixRowWritable>{
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			if(path.indexOf("/IV")>-1){	//Importance vectors
				outKey.set(Long.parseLong(items[0]), 1);
				double[] vec = new double[items.length-1];
				for(int i=0;i<items.length-1;i++)
					vec[i] = Double.parseDouble(items[i+1]);
				outValue.set(vec);
			} else if(path.indexOf("/NumLinks")>-1){
				outKey.set(Long.parseLong(items[0]), 0);
				outValue.set(Double.parseDouble(items[1]));
			} else {
				items = value.toString().split("\t");
				outKey.set(Long.parseLong(items[0]), 2);
				outValue.set(Long.parseLong(items[1]),1.0);
			}
			context.write(outKey, outValue);
		}
	}
	
	public static class FirstPartitioner extends Partitioner<LongPairWritable, MatrixRowWritable> {
		
		HashPartitioner<Long, MatrixRowWritable> p = new HashPartitioner<Long, MatrixRowWritable>();
		@Override
		public int getPartition(LongPairWritable key, MatrixRowWritable value, int numPartitions) {
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
	 * Reducer for distributing data with pair
	 * @author tigerzhong
	 *
	 */
	public static class RWDisPairReducer extends Reducer<LongPairWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private double r = 0;
		
		private double alpha = 0.9;
		
		@Override
		public void reduce(LongPairWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			double norm = 0.0;
			double[] iv = null;
			double[] outV = null;;
			for(MatrixRowWritable item : values){
				if(key.getSecond()==0L){
					norm = item.getFirstVal();
				} else if(key.getSecond()==1L) {
					iv = item.viewVector();
					outV = new double[iv.length];
				} else {
					long id = item.getFirstID();
					double dist = item.getFirstVal();
		  			outKey.set(id);
		  			for(int i=0;i<iv.length;i++) outV[i] = iv[i]*dist/norm*alpha;
		  			outValue.set(outV);
		  			context.write(outKey, outValue);
				}
			}
			for(int i=0;i<iv.length;i++) iv[i] = r*(1-alpha);
	  		outValue.set(iv);
	  		outKey.set(key.getFirst());
	  		context.write(outKey, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			r = 1.0/context.getConfiguration().getInt("model.numNode", 1);
			alpha = context.getConfiguration().getFloat("model.alpha", (float) 0.9);
		}
		
	}
}
