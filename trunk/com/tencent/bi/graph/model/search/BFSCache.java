package com.tencent.bi.graph.model.search;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;

/**
 * Breath First Search
 * @author tigerzhong
 *
 */
public class BFSCache {
	/**
	 * Number of iterations
	 */
	protected int numIt = 100;
	/**
	 * Input path
	 */
	protected String inputPath = "";
	
	protected int lower_bound, upper_bound;
	
	public void initModel(int numIt, int lower_bound, int upper_bound, String inputPath){
		this.numIt = numIt;
		this.inputPath = inputPath;
		this.lower_bound = lower_bound;
		this.upper_bound = upper_bound;
	}
	
	/**
	 * Perform search
	 * @throws Exception 
	 */
	public void performSearch() throws Exception{
		for(int it=0;it<numIt;it++){
			Configuration conf = FileOperators.getConfiguration();
			conf.setInt("data.lower.bound", lower_bound);
			conf.setInt("data.upper.bound", upper_bound);
			conf.set("data.dict.path",conf.get("hadoop.cache.path")+"BFSCache/"+it+"/");
			//First MR, flooding
			Job job = new Job(conf);
			job.setJarByClass(BFSCache.class);
			job.setJobName("BFS-It-"+it);	
			job.setMapOutputKeyClass(LongWritable.class);
			job.setMapOutputValueClass(NullWritable.class);
			job.setOutputKeyClass(LongWritable.class);
			job.setOutputValueClass(NullWritable.class);
			job.setMapperClass(BFSMapper.class);
			job.setReducerClass(BFSReducer.class);
			job.setInputFormatClass(TextInputFormat.class);
			job.setOutputFormatClass(TextOutputFormat.class);
			FileInputFormat.addInputPath(job, new Path(inputPath));
			FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.cache.path")+"BFSCache/"+(it+1)+"/"));
			job.waitForCompletion(true);
		}
	}
	
	public static class BFSMapper extends Mapper<LongWritable, Text, LongWritable, NullWritable>{
		
		private LongWritable outKey = new LongWritable();
		
		private Map<Long, Boolean> dict = null;
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] pair = value.toString().split("\t");
			long k = Long.parseLong(pair[0]);
			if(!dict.containsKey(k)) return;
			outKey.set(Long.parseLong(pair[1]));
			context.write(outKey, NullWritable.get());
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			String path = conf.get("data.dict.path");
			List<String> list = DataOperators.readTextFromHDFS(conf, path);
			dict = new HashMap<Long, Boolean>(list.size());
			for(String item : list) dict.put(Long.parseLong(item), true);
			System.out.println(dict.size());
		}
		
		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException{
			super.cleanup(context);
			for(Long k : dict.keySet()){
				outKey.set(k);
				context.write(outKey, NullWritable.get());
			}
		}
	}
	
	/**
	 * For flooding from queued nodes to others
	 * @author tigerzhong
	 *
	 */
	public static class BFSReducer extends Reducer<LongWritable, NullWritable, LongWritable, NullWritable>{
		
		private int lower_bound, upper_bound;
		
		@Override
		public void reduce(LongWritable key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
			int numValue = 0;
			for(@SuppressWarnings("unused") NullWritable value : values) {
				numValue ++;
				if(numValue>upper_bound) return;
			}
			if(numValue<lower_bound) return;
			context.write(key, NullWritable.get());
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			Configuration conf = context.getConfiguration();
			lower_bound = conf.getInt("data.lower.bound", 10);
			upper_bound = conf.getInt("data.upper.bound", 10);
		}
	}
}