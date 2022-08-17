# 1 - DEVEMOS ADICIONAR UM INPUT PARA DEFINIR QUAL O TIPO DE PERDA A SER UTILIZADA PARA ENCONTRAR O CLASSIFICADOR
# NA FASE DE VALIDACAO E CONSEQUENTEMENTE AVALIACAO NO TESTE

# 2 - SUBSTITUIR PREDITO VS OBSERVADO POR OBSERVADO VS G(X) E PONTO DE CORTE PARA CLASSIFICACAO

# 3 - SUBSTITUIR EQM POR MEDIDAS DE CLASSFIICACAO E ADICIONAR AS TABELAS CRUZADAS (OBSERVADO E PREDITO).

# COLOCAR AUTOMATICO PARA 0.5 A CLASSIFICACAO, MAS ESTE PONTO DE CORTE COMO UM SELECTINPUT, TALVEZ UM SELECTINPUT
# PARA CADA FUNCAO

moedor_clas <- function(dados,
                       target,
                       input_clas,
                       p,
                       modelos){
  # set.seed(1)
  # if(!(dados[,target] %>% is.numeric)){
  #   stop("A variável Target deve ser numérica")
  # }
  
  # 
  # Obter vetor da Target, observe que somente vai funcionar como classificador binario
  x <- dados
  x[,names(x) == target] <- x[,names(x) == target] %>% as.factor
  
  # Obter a matriz de planejamento
  x <-  model.matrix(~.-1, data = x[,names(x) %in% c(target,input_reg)] ) %>% as.matrix 
  
  # Separar target e input. Centralizar e padronizar escala dos inputs.
  y <- x[,colnames(x) == target] %>% as.numeric
  x <- x[,colnames(x) != target] %>% apply(2, scale, center=T,scale=T) %>% as.matrix
  # 
  
  #   
  #   # Obter vetor da Target
  # y <- dados[,names(dados) == target] %>% as.numeric
  # # Obter a matriz Input
  # x <-  model.matrix(~.-1, data = dados[,names(dados) %in% c(input_reg)] ) %>% 
  #   apply(2, scale, center=T,scale=T) %>% 
  #   as.matrix
  # 
  
  # p entre 0 e 1
  split <-  sample(c("Treinamento",
                     "Teste"),prob=c(p,1-p),size = length(y),
                   replace = TRUE)
  
  aux <- data.frame(y_teste = y[split=="Teste"])
  
  if( ("MMQ" %in% modelos) || ("Lasso" %in% modelos) ){
    vc_lasso  <-  cv.glmnet(x[split=="Treinamento",], y[split=="Treinamento"], alpha = 1)
    if( "MMQ" %in% modelos ){
      # MMQ
      y_mmq =  predict(vc_lasso, s = 0, newx = x[split=="Teste",])
      colnames(y_mmq) = "MMQ"
      aux <- data.frame(aux, 
                        y_mmq )
    }
    if( "Lasso" %in% modelos ){
      # Lasso
      y_lasso =  predict(vc_lasso, s = vc_lasso$lambda.min, newx = x[split=="Teste",])
      colnames(y_lasso) = "Lasso"
      aux <- data.frame(aux, 
                        y_lasso )
    }
  }
  
  if( "Ridge" %in% modelos ){  
    # Ridge
    vc_ridge  <-  cv.glmnet(x[split=="Treinamento",], y[split=="Treinamento"], alpha = 0)
    y_ridge = predict(vc_ridge, s = vc_ridge$lambda.min, newx = x[split=="Teste",])
    colnames(y_ridge) = "Ridge"
    aux <- data.frame(aux, 
                      y_ridge )
  }
  
  if( "Floresta" %in% modelos ){  
    # floresta aleatória, talvez colocar numero de arvores como parametro
    floresta <- randomForest(x[split=="Treinamento",], y[split=="Treinamento"])
    aux <- data.frame(aux, 
                      Floresta = predict(floresta, x[split=="Teste",]) )
  }
  return(aux)
}
