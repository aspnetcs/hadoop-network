package com.tencent.bi.cf.optimization.gradient.common;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

import com.tencent.bi.utils.serialization.LongPairWritable;

public class LongPartition {
	
	public static class ModLongPairPartitioner extends Partitioner<LongPairWritable, Writable> {
		
		HashPartitioner<Long, Writable> p = new HashPartitioner<Long, Writable>();
		@Override
		public int getPartition(LongPairWritable key, Writable value, int numPartitions) {
			return p.getPartition(key.getFirst()/100, value, numPartitions);
		}
		
	}
	
	public static class ModLongPartitioner extends Partitioner<LongWritable, Writable> {
		
		HashPartitioner<Long, Writable> p = new HashPartitioner<Long, Writable>();
		@Override
		public int getPartition(LongWritable key, Writable value, int numPartitions) {
			return p.getPartition(key.get()/100, value, numPartitions);
		}
		
	}
	
}
