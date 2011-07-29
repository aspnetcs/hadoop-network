package com.tencent.bi.utils.hadoop;

import java.io.IOException;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;

import com.tencent.bi.cf.optimization.Loss;

public class ModelOperator {
	
	public static String loss2String(Configuration conf, Loss func) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<Loss> output = new DefaultStringifier<Loss>(conf, Loss.class); 
		String str = output.toString(func);
		return str;
	}
	
	public static Loss string2Loss(Configuration conf, String str) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<Loss> ds = new DefaultStringifier<Loss>(conf, Loss.class); 
		Loss l = (Loss) ds.fromString(str);
		return l;
	}

}
