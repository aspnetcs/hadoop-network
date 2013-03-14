package com.tencent.bi.driver;

//import com.tencent.bi.cf.model.naive.RelationalPredictionEnhanced;
import com.tencent.bi.cf.model.knn.RelationalPrediction;
import com.tencent.bi.graph.model.diameter.HADI;
import com.tencent.bi.graph.model.search.BFSCache;
import com.tencent.bi.graph.utility.CompositeNetwork;
import com.tencent.bi.graph.utility.DataProcess;
import com.tencent.bi.utils.StringUtils;
import com.tencent.bi.utils.hadoop.FileOperators;

//import java.io.IOException;

//import com.tencent.mf.model.naive.Bias;
//import com.tencent.bi.utils.io.MatrixIO;
//import com.tencent.bi.cf.model.common.MatrixInitialization;



public class TestDriver {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws Exception {
		
		/*// Matrix init
		int nr = Integer.parseInt(args[0]);
		int nc = Integer.parseInt(args[1]);
		String path = args[2];
		MatrixInitialization obj = new MatrixInitialization();
		obj.init(nr, nc, path, true);
		obj.perform();
		//Building mapping table
		int cid = Integer.parseInt(args[0]);
		String inName = args[1];
		String outName = args[2];
		MappingTable obj = new MappingTable();
		obj.init(cid, inName, outName);
		obj.perform();		
		Bias.getBias(args[0], Constant.TEMP_PATH+"BiasTmp/");*/
		//MatrixIO.readMatrix(args[0]);
		FileOperators.confName = args[0];
		//RelationalPredictionEnhanced model = new RelationalPredictionEnhanced();
		//model.initModel(args[0], args[1], Integer.parseInt(args[2]));
		//model.buildModel();
		if(args[1].equalsIgnoreCase("MicroBlogInc")){
			DataProcess.extractMicroBlogInc(args[2], args[3], args[4]);
		} else if(args[1].equalsIgnoreCase("CSUsrADDay")){
			DataProcess.extractCSUsrADDay(args[2], args[3], args[4], args[5]);
		} else if(args[1].equalsIgnoreCase("UserTagInc")){
			DataProcess.extractUserTagInc(args[2], args[3], args[4], args[5]);
		} else if(args[1].equalsIgnoreCase("MicroBlogWhole")){
			DataProcess.extractMicroBlogWhole(args[2], args[3], args[4]);
		} else if(args[1].equalsIgnoreCase("QQWhole")){
			DataProcess.extractQQWhole(args[2], args[3], args[4]);
		}  else if(args[1].equalsIgnoreCase("UserInfo")){
			DataProcess.extractUserInfo(args[2], args[3], args[4]);
		} else if(args[1].equalsIgnoreCase("BFS")){
			BFSCache model = new BFSCache();
			model.initModel(Integer.parseInt(args[2]), Integer.parseInt(args[3]), Integer.parseInt(args[4]), args[5]);
			model.performSearch();
		} else if(args[1].equalsIgnoreCase("HADI")){
			HADI.buildModel(args[2], args[3], Integer.parseInt(args[4]), Double.parseDouble(args[5]));
		}
		else if(args[1].equalsIgnoreCase("Bin"))
			CompositeNetwork.text2Bin(args[2], args[3], args[4]);
		else if(args[1].equalsIgnoreCase("Modify"))
			StringUtils.modifyKey(args[2], args[3]);
		else if(args[1].equalsIgnoreCase("RP")){
			RelationalPrediction model = new RelationalPrediction();
			model.initModel(args[2], args[3], Integer.parseInt(args[4]), Integer.parseInt(args[5]), Double.parseDouble(args[6]));
			model.buildModel();
		}
		else CompositeNetwork.getOverlapping(args[2], args[3], args[4]);
	}

}
