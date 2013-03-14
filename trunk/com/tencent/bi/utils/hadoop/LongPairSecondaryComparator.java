package com.tencent.bi.utils.hadoop;

import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

import com.tencent.bi.utils.serialization.LongPairWritable;

/**
 * Comparators for Secondary Sort using LongPairWritable as key
 * @author tigerzhong
 *
 */
public class LongPairSecondaryComparator {
	
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
}
