package com.tencent.bi.graph.utility;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
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
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.hadoop.FileOperators;
import com.tencent.bi.utils.hadoop.LongPairSecondaryComparator;
import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class CompositeNetwork {

	public static void getOverlapping(String mapPath, String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		conf.set("data.map.path", mapPath);
		//Filtering A
		conf.setBoolean("data.idx", false);
		Job job = new Job(conf);
		job.setJarByClass(CompositeNetwork.class);
		job.setJobName("Get-Overlapping-IM-MicroBlog");	
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(NullWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(FilterMapper.class);
		job.setPartitionerClass(FirstPartitioner.class);
		job.setGroupingComparatorClass(LongPairSecondaryComparator.GroupComparator.class);
		job.setSortComparatorClass(LongPairSecondaryComparator.KeyComparator.class);
		job.setReducerClass(FilterReducer.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileInputFormat.addInputPath(job, new Path(mapPath));
		FileOutputFormat.setOutputPath(job, new Path(conf.get("hadoop.tmp.path")+"overlap/"));
		job.waitForCompletion(true);	
		//Filtering B
		conf.setBoolean("data.idx", true);
		job = new Job(conf);
		job.setJarByClass(CompositeNetwork.class);
		job.setJobName("Get-Overlapping-IM-MicroBlog");	
		job.setMapOutputKeyClass(LongPairWritable.class);
		job.setMapOutputValueClass(NullWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(FilterMapper.class);
		job.setPartitionerClass(FirstPartitioner.class);
		job.setGroupingComparatorClass(LongPairSecondaryComparator.GroupComparator.class);
		job.setSortComparatorClass(LongPairSecondaryComparator.KeyComparator.class);
		job.setReducerClass(FilterReducer.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(conf.get("hadoop.tmp.path")+"overlap/"));
		FileInputFormat.addInputPath(job, new Path(mapPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);	
	}
	
	public static class FilterMapper extends Mapper<LongWritable, LongWritable, LongPairWritable, NullWritable>{
		
		private LongPairWritable outKey = new LongPairWritable();
		
		private boolean idx = false;
		
		private String mPath = "";
		
		@Override
		public void map(LongWritable key, LongWritable value, Context context) throws IOException, InterruptedException {
			String path = ((FileSplit) context.getInputSplit()).getPath().toString();
			if(path.indexOf(mPath)>-1) outKey.set(key.get(), -1);
			else {
				if(!idx) outKey.set(key.get(), value.get());
				else outKey.set(value.get(), key.get());
				context.write(outKey, NullWritable.get());
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getBoolean("data.idx", false);
			mPath = context.getConfiguration().get("data.map.path");
		}
	}
	
	public static class FirstPartitioner extends Partitioner<LongPairWritable, NullWritable> {
		
		HashPartitioner<Long, NullWritable> p = new HashPartitioner<Long, NullWritable>();
		@Override
		public int getPartition(LongPairWritable key, NullWritable value, int numPartitions) {
			return p.getPartition(key.getFirst(), value, numPartitions);
		}
		
	}
	
	public static class FilterReducer extends Reducer<LongPairWritable, NullWritable, LongWritable, LongWritable>{
		
		LongWritable outKey = new LongWritable();
		
		LongWritable outValue = new LongWritable();
		
		boolean idx = false;
		
		@Override
		public void reduce(LongPairWritable key, Iterable<NullWritable> values, Context context) throws IOException, InterruptedException {
	  		boolean flag = false;
			if(key.getSecond()==-1) flag=true;
			else {
				if(flag){
					if(!idx) {
						outKey.set(key.getFirst());
						outValue.set(key.getSecond());
					} else {
						outKey.set(key.getSecond());
						outValue.set(key.getFirst());
					}
					context.write(outKey, outValue);
				}
			}
		}
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException{
			super.setup(context);
			idx = context.getConfiguration().getBoolean("data.idx", false);
		}
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	public static void generateDict(String inputPath, String outputPath) throws Exception {
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(CompositeNetwork.class);
		job.setJobName("Dictionary-Generator");	
		job.setMapOutputKeyClass(LongWritable.class);
		job.setMapOutputValueClass(BooleanWritable.class);
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		job.setMapperClass(DictMapper.class);
		job.setCombinerClass(DictCombiner.class);
		job.setReducerClass(DictReducer.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);	
	}
	
	public static class DictMapper extends Mapper<LongWritable, LongWritable, LongWritable, BooleanWritable>{
		
		BooleanWritable outValue = new BooleanWritable(true);
		
		@Override
		public void map(LongWritable key, LongWritable value, Context context) throws IOException, InterruptedException {
			context.write(key, outValue);
			context.write(value, outValue);
		}
	}

	public static class DictCombiner extends Reducer<LongWritable, BooleanWritable, LongWritable, BooleanWritable>{
		
		BooleanWritable outValue = new BooleanWritable(true);
		
		@Override
		public void reduce(LongWritable key, Iterable<BooleanWritable> values, Context context) throws IOException, InterruptedException {
			context.write(key, outValue);
		}
		
	}
	
	public static class DictReducer extends Reducer<LongWritable, BooleanWritable, LongWritable, LongWritable>{
		
		@Override
		public void reduce(LongWritable key, Iterable<BooleanWritable> values, Context context) throws IOException, InterruptedException {
	  		context.write(key, key);
	  		context.getCounter("Eval", "Cnt").increment(1);
		}
		
	}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	public static void text2Bin(String inputPath, String outputPath, String type) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		Job job = new Job(conf);
		job.setJarByClass(CompositeNetwork.class);
		job.setJobName("Text2Binary-"+type);	
		job.setOutputKeyClass(LongWritable.class);
		job.setOutputValueClass(LongWritable.class);
		if(type.equalsIgnoreCase("IM")) job.setMapperClass(IMTransformMapper.class);
		else if(type.equalsIgnoreCase("MicroBlog")) job.setMapperClass(MicroBlogTransformMapper.class);
		else if(type.equalsIgnoreCase("UserInfor")) {
			job.setMapperClass(UserInfoTransformMapper.class);
			job.setOutputValueClass(MatrixRowWritable.class);
		}
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		job.setInputFormatClass(TextInputFormat.class);
		SequenceFileOutputFormat.setCompressOutput(job, true);
		SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
		SequenceFileOutputFormat.setOutputCompressionType(job, CompressionType.BLOCK);
		FileInputFormat.addInputPath(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		job.waitForCompletion(true);	
	}
	
	public static class MicroBlogTransformMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{
		
		LongWritable outKey = new LongWritable();
		
		LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split("\t");
			outKey.set(Long.parseLong(items[0]));
			outValue.set(Long.parseLong(items[1]));
			context.write(outKey, outValue);
		}
	}
	
	public static class IMTransformMapper extends Mapper<LongWritable, Text, LongWritable, LongWritable>{
		
		LongWritable outKey = new LongWritable();
		
		LongWritable outValue = new LongWritable();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split(";");
			if(!items[2].split(",")[0].equalsIgnoreCase("1")) return;
			if(!Character.isDigit(items[0].charAt(0)) || !Character.isDigit(items[1].charAt(0))) return;
			outKey.set(Long.parseLong(items[0]));
			outValue.set(Long.parseLong(items[1]));
			context.write(outKey, outValue);
		}
	}
	
	public static class UserInfoTransformMapper extends Mapper<LongWritable, Text, LongWritable, MatrixRowWritable>{
		
		LongWritable outKey = new LongWritable();
		
		MatrixRowWritable outValue = new MatrixRowWritable();
		
		List<Long> idList = new ArrayList<Long>();
		
		List<Double>  valList = new ArrayList<Double>();
		
		@Override
		public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			String[] items = value.toString().split(",",-1);
			key.set(Long.parseLong(items[1]));
			for(int i=2;i<items.length;i++){
				if(items[i].equalsIgnoreCase("")) continue;
				else {
					idList.add(i-2l);
					valList.add(Double.parseDouble(items[i]));
				}
			}
			outValue.set(idList, valList, items.length-2);
			context.write(outKey, outValue);
		}
	}
}
