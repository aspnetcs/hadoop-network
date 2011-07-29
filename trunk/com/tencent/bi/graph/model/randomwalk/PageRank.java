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

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

/**
 * Page Rank or Random Walk with Restart
 * pa(i+1) = \alpha*sum_{b\in N(a)}[pb(i)/N(b)] + (1-\alpha)*pa(0)
 * @author tigerzhong
 *
 */
public class PageRank extends AbstractRWModel{
	
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
		job.setJarByClass(PageRank.class);
		job.setJobName("PageRank-Dist-It-"+iteration);	
		job.setMapOutputValueClass(MatrixRowWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(MatrixRowWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		if(isADList){	//use adjacency list to store graph
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapperClass(Mapper.class);
			job.setReducerClass(RWDisAdListReducer.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
		} else {		//use pair to store graph
			job.setMapOutputKeyClass(Text.class);
			job.setMapperClass(RWDisPairMapper.class);
			job.setPartitionerClass(FirstPartitioner.class);
			job.setSortComparatorClass(KeyComparator.class);
			job.setGroupingComparatorClass(GroupComparator.class);
			job.setReducerClass(RWDisPairReducer.class);
			job.setInputFormatClass(TextInputFormat.class);
			FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.cache.path")+"NumLinks/"));
		}
		FileInputFormat.addInputPath(job, new Path(adListPath));
		FileInputFormat.addInputPath(job, new Path(inputPath));	//Importance value vectors
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"));
		job.waitForCompletion(true);
		FileSystem fs = FileSystem.get(conf); 
		fs.delete(new Path(inputPath), true);
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
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"));
		FileOutputFormat.setOutputPath(job, new Path(inputPath));
		job.waitForCompletion(true);
		fs.delete(new Path(conf.get("hadoop.tmp.path")+"PageRank-Dist/"), true);
	}

	/**
	 * Reducer for distributing data with ad list
	 * @author tigerzhong
	 *
	 */
	public static class RWDisAdListReducer extends Reducer<LongWritable, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private double r = 0.0;
		
		@Override
		public void reduce(LongWritable key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			double[] valueList = null;	//importance value list of one node
			double[] probList = null;	//transmission probabilities of one's neighbors
			long[] probIDs = null;		//corresponding IDs
			for(MatrixRowWritable item : values){
				if(item.isSparse()){
					probList = item.getVector();
					probIDs = item.getIDs();
				}
				else valueList = item.getVector();
			}
			if(probIDs == null || valueList==null){	//something is missing...
				context.getCounter("Eval", "Missing").increment(1);
				return;				
			}
	  		int numN = probIDs.length;
	  		for(int i=0;i<numN;i++){
	  			double[] probValueList = new double[valueList.length];
	  			for(int k=0;k<probValueList.length;k++)
	  				probValueList[k] = valueList[k]*probList[i]/numN;	//V_b = V_a*P(b|a)
	  			outKey.set(probIDs[i]);
	  			outValue.set(probValueList);
	  			context.write(outKey, outValue);
	  		}
	  		long[] rowIDs = new long[valueList.length];
	  		for(int i=0;i<valueList.length;i++){
	  			valueList[i] = 1/r;
	  			rowIDs[i] = i;
	  		}
	  		outValue.set(rowIDs, valueList, valueList.length);
	  		context.write(key, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			r = 1.0/context.getConfiguration().getInt("model.numNode", 1);
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
			double[] valueList = new double[numValues];
			for(MatrixRowWritable item : values){
				double[] tmpList = item.getVector();
				if(item.isSparse()){	//Original values
					for(int i=0;i<numValues;i++)
						valueList[i]+=tmpList[i]*(1-alpha);
				}
				else {
					for(int i=0;i<numValues;i++)
						valueList[i]+=tmpList[i]*alpha;
				}
			}
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
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public static class RWDisPairMapper extends Mapper<LongWritable, Text, Text, MatrixRowWritable>{
		
		private Text outKey = new Text();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			String[] items = value.toString().split(",");
			if(path.indexOf("/IV")>-1){	//Importance vectors
				outKey.set(items[0]+",1");
				double[] vec = new double[items.length-1];
				for(int i=0;i<items.length-1;i++)
					vec[i] = Double.parseDouble(items[i+1]);
				outValue.set(vec);
			} else if(path.indexOf("/NumLinks")>-1){
				outKey.set(items[0]+",0");
				outValue.set(Double.parseDouble(items[1]));
			} else {
				items = value.toString().split("\t");
				outKey.set(items[0]+",2");
				outValue.set(Long.parseLong(items[1]),1.0);
			}
			context.write(outKey, outValue);
		}
	}
	
	public static class FirstPartitioner extends Partitioner<Text, MatrixRowWritable> {

		@Override
		public int getPartition(Text key, MatrixRowWritable value, int numPartitions) {
			long k = Long.parseLong(key.toString().split(",")[0]);
			return Math.abs((int)k * 127) % numPartitions;
		}
	}
	
	public static class KeyComparator extends WritableComparator {
	    
		protected KeyComparator() {
	    	super(Text.class, true);
	    }
	    
	    @SuppressWarnings("rawtypes")
		@Override
	    public int compare(WritableComparable w1, WritableComparable w2) {
	    	String[] ip1 = ((Text) w1).toString().split(",");
	    	String[] ip2 = ((Text) w2).toString().split(",");
	    	int cmp = ip1[0].compareTo(ip2[0]);
			if (cmp != 0) return cmp;
			return ip1[1].compareTo(ip2[1]);
	    }
	}
	
	 public static class GroupComparator extends WritableComparator {
		 protected GroupComparator() {
			 super(Text.class, true);
		 }
		 
		 @SuppressWarnings("rawtypes")
		 @Override
		 public int compare(WritableComparable w1, WritableComparable w2) {
			 String[] ip1 = ((Text) w1).toString().split(",");
			 String[] ip2 = ((Text) w2).toString().split(",");
			 return ip1[0].compareTo(ip2[0]);
		 }
	 }
	
	/**
	 * Reducer for distributing data with pair
	 * @author tigerzhong
	 *
	 */
	public static class RWDisPairReducer extends Reducer<Text, MatrixRowWritable, LongWritable, MatrixRowWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private MatrixRowWritable outValue = new MatrixRowWritable();
		
		private double r = 0;
		
		@Override
		public void reduce(Text key, Iterable<MatrixRowWritable> values, Context context) throws IOException, InterruptedException {
			int pt = 0;
			double norm = 0.0;
			double[] iv = null;
			double[] outV = null;
			for(MatrixRowWritable item : values){
				if(pt==0){
					norm = item.getFirstVal();
					pt ++;
				} else if(pt==1) {
					iv = item.getVector();
					outV = new double[iv.length];
					pt ++;
				} else {
					long id = item.getFirstID();
					double dist = item.getFirstVal();
		  			outKey.set(id);
		  			for(int i=0;i<iv.length;i++) outV[i] = iv[i]*dist/norm;
		  			outValue.set(outV);
		  			context.write(outKey, outValue);
				}
			}
			for(int i=0;i<iv.length;i++) iv[i] = r;
	  		outValue.set(iv);
	  		outKey.set(Long.parseLong(key.toString().split(",")[0]));
	  		context.write(outKey, outValue);
		}

		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			r = 1.0/context.getConfiguration().getInt("model.numNode", 1);
		}
		
	}
}
