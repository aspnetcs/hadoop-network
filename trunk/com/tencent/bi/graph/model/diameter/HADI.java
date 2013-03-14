package com.tencent.bi.graph.model.diameter;

import org.apache.hadoop.conf.Configuration;

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.hadoop.FileOperators;

public class HADI {
	
	public static int diameter = 0;
	
	public static void buildModel(String relationPath, String outputPath, int maxIt, double eps) throws Exception{
		Configuration conf = FileOperators.getConfiguration();
		BitmaskCreator.create(relationPath, conf.get("hadoop.tmp.path")+"0/stage2/");
		double N = 0;
		for(int i=0;i<maxIt;i++){
			Stage1.perform(relationPath, conf.get("hadoop.tmp.path")+i+"/stage2/", conf.get("hadoop.tmp.path")+(i+1)+"/stage1/");
			Stage2.perform(conf.get("hadoop.tmp.path")+(i+1)+"/stage1/", conf.get("hadoop.tmp.path")+(i+1)+"/stage2/");
			Stage3.perform(conf.get("hadoop.tmp.path")+(i+1)+"/stage2/", conf.get("hadoop.tmp.path")+(i+1)+"/stage3/");
			double currentN = Double.parseDouble(DataOperators.readTextFromHDFS(conf, conf.get("hadoop.tmp.path")+(i+1)+"/stage3/").get(0).split("\t")[1]);
			if(N*(1+eps)>currentN) {
				diameter = (i+1);
				break;
			} else N = currentN;
		}
	}
	
}
