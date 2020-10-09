//Make some assumptions
// let's assume seq_len = 64 for now
// N d_mod d_ff h dk dv 
// 6 512   2048 8 64 64
// 1 12288 49152 96 128 128
Constant Seq_Len 2048;

Network Transformer {
	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


}

	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


	//Start encoder
	Layer MH_FC_DimReduce_VKQ_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// k = 3 * 12288, one for each VKQ
		Dimensions { N: Seq_Len, K: 36854, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

  // SD layers should be multiplied by head count
	Layer SD_MatMul_QK_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(dv)xN(seql)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: 128, S: 1, Y:128, X:Seq_Len }
		Dataflow {
			SpatialMap(2,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			
		}
	}

  // SD layers should be multiplied by head count
	Layer SD_MatMul_V_00 { //Mat mul, batch is 1
		Type: CONV //MatMul -> M(seql)xK(seql)xN(dv)-> filter = Kx1(m chans), input = KxN 
		Stride { X: 1, Y: 1 }
		//N=1, K(conv)=M(matr), C=1, R(conv)=K(matr),S=1,Y(conv)=K(matr), X(conv)=N(matr)
		Dimensions { N: 1, K: Seq_Len, C: 1, R: Seq_Len, S: 1, Y:Seq_Len, X:128 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	}
	
	// done with h parallel sd layers now
	Layer MH_FC_DimRecast_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV
		Stride { X: 1, Y: 1 }	// v,k,q have been combined
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_A_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV //2048 output neurons, 12288 -> 49152
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 49152, C: 1, R: 1, S: 12288, Y:1, X:12288 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good

	/// done with h parallel sd layers now
	Layer FF_B_0 { //Batched FC layer, where seq_len is batch, input is d_model x 1
		Type: CONV // 49152 -> 512
		Stride { X: 1, Y: 1 }	
		Dimensions { N: Seq_Len, K: 12288, C: 1, R: 1, S: 49152, Y:1, X:49152 }
		Dataflow {
			SpatialMap(1,1) K;
			TemporalMap(1,1) C;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;	
			Cluster(1, P);
			SpatialMap(1,1) C;
			TemporalMap(Sz(R),1) Y;
			TemporalMap(Sz(S),1) X;
			TemporalMap(Sz(R),Sz(R)) R;
			TemporalMap(Sz(S),Sz(S)) S;
		}
	} //good


}
