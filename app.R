# Dependencias
library(shiny)
library(shinydashboard)
library(rio)
library(foreign)
library(readr)
library(feather)
library(fst)
library(rmatio)
library(xml2)
library(dplyr)
library(magrittr)
library(corrplot)
library(GGally)
library(plotly)
library(ggplot2)
# library(Hmisc)
library(datasets)
library(glmnet) # MMQ, Lasso, Ridge, logistico
library(e1071) # SVM
library(randomForest) # Florestas
library(xgboost) # XGBoost
library(gridExtra)
library(crayon)
library(tidyr)
library(esquisse)
library(DALEX)

# Funções e bases ----

# Base dados simulada
x1 <- rnorm(2000) %>% round(digits = 4)
x2 <- rexp(2000) %>% round(digits = 4)
x3 <- rexp(2000) %>% round(digits = 4)
x4 <- rexp(2000) %>% round(digits = 4)
x5 <- rexp(2000) %>% round(digits = 4)
e <- rnorm(2000,0,0.5)
y <- (x1^2 + x2 + e) %>% round(digits = 4)
DB_joao <- data.frame(y,x1,x2,x3,x4,x5)
y2 <- (x1^2 + x1^3 + x1^4 + x1^5 + e) %>% round(digits = 4)
DB2_joao <- data.frame(y = y2,x1,x2,x3,x4,x5)
y3 <- (x1^2 + ifelse(x1 > 0, x2, x3) + e) %>% round(digits = 4)
DB3_joao <- data.frame(y = y3,x1,x2,x3,x4,x5)

# Funcao transforma dataset em matrix, compila modelos de regressao
# veja https://pbiecek.github.io/DALEX_docs/2-1-explainFunction.html

model_performance <- function(explainer, ...) {
  if (!("explainer" %in% class(explainer))) stop("The model_performance() function requires an object created with explain() function.")
  if (is.null(explainer$data)) stop("The model_performance() function requires explainers created with specified 'data' parameter.")
  if (is.null(explainer$y)) stop("The model_performance() function requires explainers created with specified 'y' parameter.")
  
  observed <- explainer$y
  predicted <- explainer$predict_function(explainer$model, explainer$data, ...)
  residuals <- data.frame(predicted, observed, diff = predicted - observed)
  names(residuals) <- c("predicted", "observed", "diff")
  class(residuals) <- c("model_performance_explainer", "data.frame")
  residuals$label <- explainer$label
  residuals
}

moedor_reg <- function(dados,
                       target,
                       input_reg,
                       p,
                       modelos){
  set.seed(1)
  # if(!(dados[,target] %>% is.numeric)){
  #   stop("A variável Target deve ser numérica")
  # }
  
  # 
  # Obter vetor da Target
  x <- dados
  x[,names(x) == target] <- x[,names(x) == target] %>% as.numeric
  
  # Obter a matriz de planejamento
  suppressMessages({
    x <-  model.matrix(~.-1, data = x[,names(x) %in% c(target,input_reg)] ) %>% as.matrix 
  })
  # Separar target e input. Centralizar e padronizar escala dos inputs.
  y <- x[,colnames(x) == target] %>% as.numeric
  
  if(ncol(x) == 2){
    nome_coluna <- colnames(x)[colnames(x) != target]
    # x <- x[,colnames(x) != target] %>% scale(center=T,scale=T) %>% matrix(ncol=1)
    # colnames(x) <- nome_coluna
    x <- x[,colnames(x) != target] %>% scale(center=T,scale=T) %>% matrix(ncol=1)
    x <- cbind(1,x) %>% as.matrix
    colnames(x) <- c("intercepto",nome_coluna)
  }else{
    x <- x[,colnames(x) != target] %>% apply(2, scale, center=T,scale=T) %>% as.matrix
  }
  
  # p entre 0 e 1
  split <-  sample(c("Treinamento",
                     "Teste"),prob=c(p,1-p),size = length(y),
                   replace = TRUE)
  
  aux <- data.frame(y_teste = y[split=="Teste"])
  temp <- NULL
  var_imp <- NULL
  var_mp <- NULL
  if( "MMQ" %in% modelos & ncol(x) > 1){
    # MMQ
    temp_aux <- system.time({ 
      mmq  <-  glmnet(x[split=="Treinamento",] %>% as.matrix, y[split=="Treinamento"], alpha = 0, lambda = 0)
    })[3]
    temp <- c(temp,temp_aux)
    y_mmq <- predict(mmq, newx = x[split=="Teste",] %>% as.matrix)
    
    explainer_mmq <- DALEX::explain(mmq, data = x[split=="Teste",] %>% as.matrix, 
                                    y = y[split=="Teste"], label = "MMQ")
    vi_mmq <- variable_importance(explainer_mmq, loss_function = loss_root_mean_square)
    mp_mmq <- model_performance(explainer_mmq)
    
    var_mp <- bind_rows(var_mp, mp_mmq)
    var_imp <- bind_rows(var_imp, vi_mmq)
    
    colnames(y_mmq) = "MMQ"
    aux <- data.frame(aux, 
                      y_mmq )
  }
  
  if( "Lasso" %in% modelos & ncol(x) > 1){
    # Lasso
    temp_aux <- system.time({ 
      vc_lasso  <-  cv.glmnet(x[split=="Treinamento",] %>% as.matrix, y[split=="Treinamento"], alpha = 1)
    })[3] 
    temp <- c(temp, temp_aux)
    y_lasso =  predict(vc_lasso, s = vc_lasso$lambda.min, newx = x[split=="Teste",] %>% as.matrix)
    
    explainer_lasso <- DALEX::explain(vc_lasso, data = x[split=="Teste",] %>% as.matrix, 
                                      y = y[split=="Teste"], label = "Lasso")
    vi_lasso <- variable_importance(explainer_lasso, loss_function = loss_root_mean_square)
    mp_lasso <- model_performance(explainer_lasso)
    
    var_mp <- bind_rows(var_mp, mp_lasso)
    var_imp <- bind_rows(var_imp, vi_lasso)
    
    colnames(y_lasso) = "Lasso"
    aux <- data.frame(aux, 
                      y_lasso )
  }
  
  if( "Ridge" %in% modelos & ncol(x) > 1 ){  
    # Ridge
    temp_aux <- system.time({ 
      vc_ridge  <-  cv.glmnet(x[split=="Treinamento",] %>% as.matrix, y[split=="Treinamento"], alpha = 0)
    })[3] 
    temp <- c(temp,temp_aux)
    y_ridge = predict(vc_ridge, s = vc_ridge$lambda.min, newx = x[split=="Teste",] %>% as.matrix)
    
    explainer_ridge <- DALEX::explain(vc_ridge, data = x[split=="Teste",] %>% as.matrix, 
                                      y = y[split=="Teste"], label = "Ridge")
    vi_ridge <- variable_importance(explainer_ridge, loss_function = loss_root_mean_square)
    mp_ridge <- model_performance(explainer_ridge)
    
    var_mp <- bind_rows(var_mp, mp_ridge)
    var_imp <- bind_rows(var_imp, vi_ridge)
    
    colnames(y_ridge) = "Ridge"
    aux <- data.frame(aux, 
                      y_ridge )
  }
  
  if( "Floresta" %in% modelos ){  
    # floresta aleatória, talvez colocar numero de arvores como parametro
    temp_aux <- system.time({ 
      floresta <- randomForest(x[split=="Treinamento",] %>% as.matrix, y[split=="Treinamento"])
    })[3]
    temp <- c(temp,temp_aux)
    
    explainer_floresta <- DALEX::explain(floresta, data = x[split=="Teste",] %>% as.matrix, 
                                         y = y[split=="Teste"], label = "Floresta")
    vi_floresta <- variable_importance(explainer_floresta, loss_function = loss_root_mean_square)
    mp_floresta <- model_performance(explainer_floresta)
    
    var_mp <- bind_rows(var_mp, mp_floresta)
    var_imp <- bind_rows(var_imp, vi_floresta)
    
    aux <- data.frame(aux, 
                      Floresta = predict(floresta, x[split=="Teste",] %>% as.matrix) )
  }
  
  if( "SVM" %in% modelos ){  
    temp_aux <- system.time({ 
      SVM <- e1071::svm(x=x[split=="Treinamento",] %>% as.matrix, y=y[split=="Treinamento"])
    })[3] 
    temp <- c(temp,temp_aux)
    
    explainer_SVM <- DALEX::explain(SVM, data = x[split=="Teste",] %>% as.matrix, 
                                    y = y[split=="Teste"], label = "SVM")
    vi_SVM <- variable_importance(explainer_SVM, loss_function = loss_root_mean_square)
    mp_SVM <- model_performance(explainer_SVM)
    
    var_mp <- bind_rows(var_mp, mp_SVM)
    var_imp <- bind_rows(var_imp, vi_SVM)
    
    aux <- data.frame(aux, 
                      SVM=predict(SVM, x[split=="Teste",] %>% as.matrix) )
  }
  
  
  if( "XGBoost" %in% modelos ){  
    temp_aux <- system.time({ 
      xgb_params <- list(colsample_bytree = 0.7, #how many variables to consider for each tree
                         subsample = 0.7, #how much of the data to use for each tree
                         booster = "gbtree",
                         max_depth = 5, #how many levels in the tree
                         eta = 0.1, #shrinkage rate to control overfitting through conservative approach
                         eval_metric = "rmse", 
                         objective = "reg:linear",
                         gamma = 0)
      bst <- xgb.train(
        data =      xgb.DMatrix(x[split=="Treinamento",] %>% as.matrix,label = y[split=="Treinamento"]),
        params = xgb_params,
        nrounds = 200)
    })[3] 
    temp <- c(temp,temp_aux)
    
    explainer_bst <- DALEX::explain( bst, data = x[split=="Teste",] %>% as.matrix, 
                                     y = y[split=="Teste"], label = "XGBoost")
    vi_bst <- variable_importance(explainer_bst, loss_function = loss_root_mean_square)
    mp_bst <- model_performance(explainer_bst)
    
    var_mp <- bind_rows(var_mp, mp_bst)
    var_imp <- bind_rows(var_imp, vi_bst)
    
    aux <- data.frame(aux, 
                      XGBoost=predict(bst, x[split=="Teste",] %>% as.matrix) )
  }
  tempo <- data.frame(Modelos = names(aux)[-1], `Tempo decorrido` = temp)
  return(list(aux, tempo, var_imp, var_mp))
}

## Header ----
header <-   shinydashboard::dashboardHeader( title = "QuickAnalytics", titleWidth = 600)

## Sidebar Menu ----
sidebar <- shinydashboard::dashboardSidebar(collapsed = FALSE, 
                                            sidebarMenu(
                                              menuItem("Página inicial", tabName = "inicial", icon = icon("home"),selected = TRUE),
                                              menuItem("Dados (importar e exportar)", tabName = "imp_exp", icon = icon("exchange")),
                                              menuItem("Ambiente Gráfico", tabName = "graficos", icon = icon("chart-area")),
                                              menuItem("Descritiva", icon = icon("calculator"),
                                                       menuSubItem("Variável numérica", tabName = "var_num", icon = icon("chart-bar")),
                                                       menuSubItem("Variável categórica", tabName = "var_cat", icon = icon("chart-pie")),
                                                       menuSubItem("Correlações", tabName = "desc_mult", icon = icon("chart-line"))),
                                              # menuItem("Modelos", icon = icon("unlock"),
                                              #menuSubItem("Classificação", tabName = "mod_class", icon = icon("lock")),
                                              # menuSubItem("Não supervisionado", tabName = "mod_naosuperv", icon = icon("lock")),
                                              menuItem("Regressão", tabName = "mod_reg", icon = icon(name="code-branch", class = NULL, lib = "font-awesome") ) ,
                                              menuItem("Tutoriais", icon = icon("book"),
                                                       menuSubItem( a("SAS vs R ", icon("book-open"), href="http://rpubs.com/joao_flavio/SAS_vs_R_tratamento_de_dados",  target="_blank"), tabName = "SAS_R_dados"),
                                                       menuSubItem( a("Teste de hipóteses", icon("book-open"), href="http://rpubs.com/joao_flavio/558616",  target="_blank"), tabName = "SAS_R_dados")),
                                              menuItem("Sobre", tabName = "sobre", icon = icon("user"))
                                            )
)

## Dashboard Body ----
body <- shinydashboard::dashboardBody(
  ## Itens ----
  tabItems(
    
    ## Pagina inicial ----
    tabItem(tabName = "inicial",
            fluidRow(
              column(width = 1),
              column(width = 8, includeMarkdown("./welcome.Rmd")),
              column(width = 3)
            )
    ), # Pagina inicial fechada
    
    ## Dados (importar exportar) ----
    tabItem(tabName = "imp_exp",
            # Coluna Lateral abre
            column(width = 3,
                   # Caixa abre
                   box(width = NULL, height = NULL, 
                       # Input: Selecionar numero de linhas para apresentar no display
                       radioButtons("visu_dados", "Carregar base de dados", 
                                    choices = c("PC" = "carr_pc", "Servidor" = "carr_serv"), selected = "carr_serv"),
                       # Linha horizontal
                       tags$hr(),
                       # Painel condicional para o tipo de carregamento
                       conditionalPanel(
                         condition = "input.visu_dados == 'carr_pc'",
                         # Input: Selecionar arquivo
                         fileInput("import", "Selecionar arquivo (importar)", multiple = TRUE, 
                                   buttonLabel = "Abrir...", placeholder = "25 mb no máximo")
                       ),
                       
                       # Painel condicional para o tipo de carregamento
                       conditionalPanel(
                         condition = "input.visu_dados == 'carr_serv'",
                         # Bases dados para carregar
                         uiOutput("bases_disponiveis")
                       ),
                       
                       # Botão aciona visualização dos dados
                       actionButton("atual_visu", "Carregar", icon = icon("upload")),
                       # Linha horizontal
                       tags$hr(),
                       # Input: Seleciona a extensão do arquivo para exportar
                       selectInput("export", "Selecionar extensão (exportar):",
                                   choices = c(".csv",	".psv",	".tsv",	".sas7bdat", ".xpt", ".sav",	".dta",	".xlsx",
                                               ".R",	".RData", ".rda", ".rds",	".dbf",	".arff",	".fwf",	".csv.gz", 
                                               ".feather", ".fst", ".json", ".mat", ".ods", ".html", ".yml")),
                       # Botão de download
                       downloadButton("dados_baixar", "Download")
                   ) # Caixa fecha
            ), # Coluna Lateral fecha
            # Coluna Principal abre 
            column(width = 9,
                   # mudar o tipo das colunas
                   coerceUI(id = "coerce"),
                   verbatimTextOutput(outputId = "print_tipos"),
                   # Caixa abre
                   box( width = NULL, height = NULL,
                        # Tabela de dados
                        div(style = 'overflow-x: scroll', dataTableOutput('dados_tab'))
                   )# Caixa fecha
            ) # Coluna Principal fecha 
    ), # Item Dados (importar exportar) fecha 
    
    # Gráficos ----
    tabItem(tabName = "graficos", 
            column(width = 12,
                   # Caixa abre 
                   box(width = NULL, height = NULL,
                       # Botão aciona atualizacao dos dados para o grafico
                       actionButton("atual_grafico", "  Atualizar", icon = icon("redo")),
                       tags$style("html, body {overflow: visible !important;"),
                       
                       tags$div(
                         style = "height: 500px;", # needs to be in fixed height container
                         esquisserUI(
                           id = "esquisse", 
                           header = FALSE, # dont display gadget title
                           choose_data = FALSE # dont display button to change data
                         )
                       )
                   )
            )
    ), # Item Gráficos
    
    # Descritiva variaveis numericas ----
    tabItem(
      tabName = "var_num",
      # Coluna Lateral abre
      column(width = 4,
             # Caixa abre
             box( title = "Variáveis",width = NULL, height = NULL, collapsible = FALSE,
                  # Variaveis dependendo dos dados carregados
                  uiOutput("vars_num")
             ),
             box( title = "Resumo",width = NULL, height = NULL, collapsible = FALSE,
                  tableOutput("resumo_num")
             )
      ), # Coluna Lateral fecha
      # Coluna Principal Abre
      column(width = 8,
             # Histograma
             tabBox( width = NULL, height = NULL,
                     tabPanel(h5("Histograma"),
                              plotlyOutput("hist_num"),
                              # Histograma
                              box( title = "Histograma personalização" , width = NULL, height = NULL, collapsible = TRUE, collapsed = TRUE,
                                   textInput(inputId = "hist_main", label = "Titulo:", value = "digite..."),
                                   textInput(inputId = "hist_xlab", label = "Rótulo eixo horizontal:", value = "digite..."),
                                   radioButtons("hist_ylab", label = "Eixo vertical", choices = list("Frequência" = "freq", "Proporção" = "prop",
                                                                                                     "Frequência acumulada" = "freq_acum"), selected = "freq")
                              )
                     ), # Histograma fecha
                     # Boxplot
                     tabPanel(h5("Boxplot"),
                              plotlyOutput("boxplot_num"),
                              # Boxplot
                              box( title = "Boxplot personalização" , width = NULL, height = NULL, collapsible = TRUE, collapsed = TRUE,
                                   textInput(inputId = "box_main", label = "Titulo:", value = "digite..."),
                                   textInput(inputId = "box_var", label = "Eixo horizontal:", value = "digite...")
                              )
                     ) # Boxplot Fecha
             )
      ) # Coluna Principal Fecha
    ), # Item Descritiva variaveis numericas fecha
    
    # Descritiva variaveis categoricas ----
    tabItem(
      tabName = "var_cat",
      # Coluna Lateral abre
      column(width = 4,
             # Caixa abre
             box( title = "Variáveis", width = NULL, height = NULL, collapsible = FALSE,
                  # Variaveis dependendo dos dados carregados
                  uiOutput("vars_cat")
             ),
             box( title = "Resumo", width = NULL, height = NULL, collapsible = FALSE,
                  tableOutput("resumo_cat")
             )
      ), # Coluna Lateral fecha
      # Coluna Principal abre
      column(width = 8,
             tabBox( width = NULL, height = NULL, 
                     
                     
                     # Barplot abre
                     tabPanel(h5("Gráfico de barras"),
                              plotlyOutput("bar_cat"), # Caixa fecha
                              # Grafico barras
                              box( title = "Gráfico de barras personalização" , width = NULL, height = NULL, collapsible = TRUE, collapsed = TRUE,
                                   textInput(inputId = "bar_main", label = "Titulo:", value = "digite..."),
                                   textInput(inputId = "bar_xlab", label = "Rótulo eixo horizontal:", value = "digite...")
                              )
                     ),# Barplot Fecha
                     # Barplot abre
                     tabPanel(h5("Gráfico de setores"),
                              plotlyOutput("set_cat"), # Histograma fecha
                              # Boxplot
                              box( title = "Gráfico de setores personalização" , width = NULL, height = NULL, collapsible = TRUE, collapsed = TRUE,
                                   textInput(inputId = "set_main", label = "Titulo:", value = "digite...")
                              ) # Boxplot Fecha
                     )# Barplot Fecha
             )
      ) # Coluna Principal fecha
    ),
    
    # Descritiva multivariada ----
    tabItem(
      tabName = "desc_mult",
      tabBox( width = NULL, height = NULL, 
              # Scaterplot abre
              tabPanel(h5("Gráfico de dispersão"),
                       plotlyOutput("disp_x_y"),
                       # Coluna Principal abre
                       column(width = 6,
                              # Gráfico de dispersão
                              box( title = "Variáveis" , width = NULL, height = NULL, collapsible = TRUE, collapsed = FALSE,
                                   uiOutput("vars_y"),
                                   uiOutput("vars_x")
                              )
                       ),
                       column(width = 6,
                              box( title = "Editar" , width = NULL, height = NULL, collapsible = TRUE, collapsed = TRUE,
                                   textInput(inputId = "disp_main", label = "Titulo:", value = "digite..."),
                                   textInput(inputId = "disp_xlab", label = "Rótulo eixo horizontal:", value = "digite..."),
                                   textInput(inputId = "disp_ylab", label = "Rótulo eixo vertical:", value = "digite...")
                              )
                       )
                       #  ) Gráfico de dispersão fecha
              ),# Scaterplot Fecha
              # Matriz correlações abre
              tabPanel(h5("Matriz de correlações"),
                       plotOutput("mat_cor"),
                       # Matriz correlações abre
                       box( title = "Configurações" , width = NULL, height = NULL, collapsible = TRUE, collapsed = FALSE,
                            column(width = 6,
                                   radioButtons("tipo_matriz", label = "Tipo de matriz", choices = list("Círculo" = "circ", "Agrupamento" = "agrup", 
                                                                                                        "Medidas (Recomendação: menos de 5 variáveis)" = "med"),
                                                selected = "circ"),
                                   actionButton("exec_mat_cor","Atualizar variáveis")
                            ),
                            column(width = 6, 
                                   actionButton("selectall_1","Des/Marcar todos") ,
                                   # Linha horizontal
                                   tags$hr(),
                                   uiOutput("vars_mat_cor")
                            )
                       ) # Matriz correlações Fecha
              ) # Matriz correlações Fecha
      )
    ),
    
    # Modelos ----
    tabItem(
      tabName = "mod_reg",
      # Coluna Lateral abre
      column(width = 2,
             # Regressão
             box( title = "Configuração" , width = NULL, height = NULL, collapsible = TRUE, collapsed = FALSE,
                  uiOutput("reg_target"),
                  sliderInput(inputId = "reg_p", label = "Proporção treinamento:", min = 0, max = 1, value = 0.7),
                  checkboxGroupInput("reg_modelos", "Selecione os modelos", 
                                     choices = c("MMQ","Lasso","Ridge","Floresta","SVM","XGBoost"), 
                                     selected = c("MMQ","Lasso","Ridge","Floresta","SVM","XGBoost") )
             ) # Regressao Fecha
      ), # Coluna Lateral fecha
      # Coluna Principal abre
      column(width = 8,
             #tabBox( width = NULL, height = NULL, 
             # Medidas abre
             box(width = NULL, height = NULL,
                 h4("Medidas de desempenho (REQM)"),
                 tableOutput("reg_med"), # Medidas Fecha
                 h4("Performance dos modelos"),
                 plotOutput("model_performance"),
                 # Dispersão obs vs pred abre
                 h4("Dispersão observado vs predito"),
                 plotOutput("reg_plot"), # Dispersão obs vs pred abre
                 # Importância das variáveis
                 h4("Importância das variáveis"),
                 plotOutput("var_imp"),
                 h4("Medidas de Importância das variaveis"),
                 verbatimTextOutput(outputId = "var_imp_out")
             ) # Importância das variáveis
             #)
             
      ), # Coluna Principal fecha
      column(width = 2,
             box(width = NULL, height = NULL,
                 actionButton("exec_reg","Executar"),
                 # Linha horizontal
                 tags$hr(),
                 actionButton("selectall_2","Des/Marcar todos") ,
                 # Linha horizontal
                 tags$hr(),
                 uiOutput("reg_input"))
      )
    ),
    
    
    # Sobre ----
    tabItem(tabName = "sobre", 
            column(width = 1),
            column(width = 11,
                   h3("Sobre o idealizador:"),
                   img(src = "joao.png"),
                   h3("João Flávio"),
                   a("Currículo Lattes ", href="http://lattes.cnpq.br/2309570131364286", icon("link"),  target="_blank"), 
                   a("Linkedin ", href="http://www.linkedin.com/in/jo%C3%A3o-fl%C3%A1vio-12a21b77/", icon("linkedin"),  target="_blank"),
                   a("Facebook ", href="http://www.facebook.com/joaoflavioas", icon("facebook"),  target="_blank"),
                   h4("Assessor em gestão de riscos e consultor estatístico"),
                   h4("Mestre em estatística pelo Programa Interinstitucional em Estatística UFSCar/USP"),
                   h4("Graduado em Estatística pela Universidade Federal de Uberlândia – UFU")
            )
    ) # Item Sobre fechado
  ) # Itens fechado
) # Dashboard Body fechado

### UI ----
ui = shinydashboard::dashboardPage(header, 
                                   sidebar, 
                                   body)

### Server ----
server = function(input, output, session) {
  # Limitamos o tamanho do arquivo para exchange em 25 mb (o padrão é 5 mb)
  options(shiny.maxRequestSize=25*1024^2)
  
  ## Dados (importar exportar) server ----
  
  # SelectInput para a base de dados disponivel  
  output$bases_disponiveis <- renderUI({
    n_data <- data()$results[ data()$results[,"Package"] == "datasets" ,"Item"]
    ind <- n_data %>% grep(" ",.)
    bases <- n_data[-ind]
    bases <- bases[sapply(bases,FUN = function(x){get(x) %>% is.data.frame()} ) ]
    bases <- c(bases,"DB_joao","DB2_joao", "DB3_joao")
    selectInput("base_disponivel", "Selecionar base interna", bases, selected = "mtcars" ) 
  })
  
  # Carregar base de dados 
  dados_origem <- eventReactive(input$atual_visu, {
    # Se carregar do Servidor
    # Se o usuario ainda nao carregou um arquivo
    if (input$visu_dados == "carr_serv" ){
      if ( is.null(input$base_disponivel) ){return(NULL)
      }else{
        database <- input$base_disponivel %>% get
        return(database)
      }       
    }
    
    # Se carregar do PC
    if (input$visu_dados == "carr_pc" ){
      if (is.null(input$import) ){return(NULL)}else{
        database = rio::import(input$import$datapath) 
        return(database)
      }       
    }
    
  }, ignoreNULL = FALSE)
  
  # Altera o tipo de coluna 
  dados <- callModule(module = coerceServer, id = "coerce", data = reactive({dados_origem()})) 
  
  # Tipo das colunas
  
  output$print_tipos <- renderPrint({
    if(is.null(dados$data)){return(cat("Nomes das colunas e seus tipos"))}
    #dados$names
    sapply(dados$data, class)
  })
  
  # Renderizar Tabela de Dados
  
  output$dados_tab <- renderDataTable({
    return(dados$data)
  })
  
  # Baixar base de dados 
  output$dados_baixar <- downloadHandler(
    filename = function() {
      paste("dados", input$export, sep = "")
    },
    content = function(file) {
      rio::export(dados$data, file)
    }
  )
  
  ## Gráficos server ----
  
  # Define uma variavel reativa
  data_r <- reactiveValues(data = NULL, name = "dados" )
  
  # Se imput$atual_grafico entao carregamos dados() em data_r$data  
  observeEvent( input$atual_grafico, {
    data_r$data <- dados$data 
    callModule( module = esquisserServer, id = "esquisse", data = data_r )
  })
  
  ## Descritiva variveis numericas server ----
  output$vars_num <- renderUI({
    if (is.null(dados$data)) return(NULL)
    itens= dados$data %>% select_if(is.numeric) %>% names
    names(itens)= itens
    selectInput("vars_num", "Selecione a variável", itens ) 
  })
  
  output$resumo_num <- renderTable({
    if (is.null(dados$data)) return(NULL)
    dres <- dados$data[,names(dados$data)==input$vars_num]
    qnt <- length(dres)
    vtr_na <- is.na(dres) 
    qnt_na <- sum(vtr_na) %>% as.integer
    dres <- dres[!vtr_na]
    qnt_dist <- dres %>% unique %>% length()
    
    data.frame(Medida = c( "Observações", "Missings", "Distintos", "Mínimo", "1 quartil", "Mediana", "Média", "3 quartil", "Máximo ", 
                           "Variância",  "Desvio padrão" , "Coeficiente de variação"   ), 
               Estimativa = c(qnt, qnt_na, qnt_dist ,summary(dres)[1:6], var(dres), sd(dres),  sd(dres)/mean(dres))
    )
    
  })
  
  output$hist_num <- renderPlotly({
    if (is.null(dados$data)) return(NULL)
    if(input$hist_main == "digite..."){main = input$vars_num}else{main = input$hist_main}
    if(input$hist_xlab == "digite..."){
      xlab <- list(    title = input$vars_num)
    }else{      xlab <- list(title = input$hist_xlab )  }
    
    x    <- dados$data[,names(dados$data)==input$vars_num]    
    
    if(input$hist_ylab == "freq"){
      ylab <- list(title = "Frequência")
      h <- plot_ly(x = x , type = "histogram") %>%
        layout( title = main, xaxis = xlab, yaxis = ylab)
    }
    
    if(input$hist_ylab == "prop"){
      ylab <- list( title = "Proporção" )
      h <- plot_ly(x = x , type = "histogram", histnorm = "probability") %>%
        layout(title = main,  xaxis = xlab, yaxis = ylab)
    }
    
    if(input$hist_ylab == "freq_acum"){
      ylab <- list(title = "Frequência")
      h <- plot_ly(x = x , type = "histogram", cumulative = list(enabled=TRUE)) %>%
        layout( title = main, xaxis = xlab, yaxis = ylab)
    }
    
    return(h)
    
  }) 
  
  output$boxplot_num <- renderPlotly({
    if (is.null(dados$data)) return(NULL)
    if(input$box_var == "digite..."){x_lab = input$vars_num}else{x_lab = input$box_var}
    if(input$box_main == "digite..."){main = "Boxplot"}else{main = input$box_main}
    dados$data[,names(dados$data)==input$vars_num] %>%
      plot_ly(y = ., type = "box", boxpoints = "all", name = x_lab) %>% 
      layout(title = main)
  }) 
  
  ## Descritiva variveis categoricas server ----
  output$vars_cat <- renderUI({
    if (is.null(dados$data)) return(NULL)
    itens= dados$data %>% select_if(function(col) is.character(col) | is.factor(col) ) %>% names
    names(itens)=itens
    selectInput("vars_cat", "Selecione a variável",itens) 
  })
  
  dados_cat <- reactive({
    if (is.null(dados$data)){
      return(NULL)
    }else{
      n <- dados$data %>% nrow
      counts <- dados$data %>% count(get(input$vars_cat))  
      names(counts) <- c("Categoria","Frequência")
      resum <- cbind(counts, "Frequência relativa" = counts$Frequência/n)
      
      if(resum$Categoria %>% is.na %>% sum >=1){
        resum[resum$Categoria %>% is.na ,1] <- "Missings"
      }
      return(resum %>% data.frame )
    }
  })
  
  
  output$resumo_cat <- renderTable({
    if (is.null(dados$data) || is.null(dados_cat()) ) return(NULL)
    resum <- dados_cat()
  })
  
  output$bar_cat <- renderPlotly({
    if (is.null(dados$data) || is.null(dados_cat()) ) return(NULL)
    if(input$bar_main == "digite..."){main = input$vars_cat}else{main = input$bar_main}
    if(input$bar_xlab == "digite..."){xlab = input$vars_cat}else{xlab = input$bar_xlab}
    
    dados_cat() %>% plot_ly(x = ~Categoria, y = ~Frequência,  type = "bar") %>% 
      layout(title= main, xaxis = list(title = xlab), yaxis = list(title = "Frequência"))
    
  }) 
  
  output$set_cat <- renderPlotly({
    if (is.null(dados$data) || is.null(dados_cat()) ) return(NULL)
    if(input$set_main == "digite..."){main = input$vars_cat}else{main = input$set_main}
    
    dados_cat() %>% plot_ly(labels = ~Categoria, values = ~Frequência) %>% 
      add_pie(hole=0.6) %>% 
      layout(title= main, showlegend = F)    
  }) 
  
  ## Descritiva multivariada server ----
  
  output$vars_x <- renderUI({
    if (is.null(dados$data)) return(NULL)
    itens=names(dados$data)
    names(itens)= itens
    selectInput("var_x", "Selecione a variável X", itens, selected = NULL ) 
  })
  
  output$vars_y <- renderUI({
    if (is.null(dados$data)) return(NULL)
    itens=names(dados$data)
    names(itens)= itens
    selectInput("var_y", "Selecione a variável Y", itens, selected = NULL ) 
  })
  
  output$disp_x_y <- renderPlotly({
    if (is.null(dados$data)) return(NULL)
    if(input$disp_main == "digite..."){main = "Dispersão" }else{main = input$disp_main}
    if(input$disp_xlab == "digite..."){xlab = input$var_x}else{xlab = input$disp_xlab}
    if(input$disp_ylab == "digite..."){ylab = input$var_y}else{ylab = input$disp_ylab}
    x    <- dados$data[,names(dados$data)==input$var_x]
    y <- dados$data[,names(dados$data)==input$var_y]
    data <- data.frame(x,y)
    plot_ly(data=data, x=~x, y=~y) %>% 
      layout(title = main,
             yaxis = list(title = ylab),
             xaxis = list(title= xlab))
  }) 
  
  output$vars_mat_cor <- renderUI({
    if (is.null(dados$data)) return(NULL)
    
    itens=names(dados$data)
    names(itens)= itens
    
    if (input$selectall_1%%2 == 0)
    {
      checkboxGroupInput("var_mat_cor", "Selecione as variáveis", itens, selected = itens ) 
    }
    else
    {
      checkboxGroupInput("var_mat_cor", "Selecione as variáveis", itens) 
    }
  })
  
  # Reativo para atualizar as variaveis (utilizando também isolate() )
  dataset_mat_cor <- eventReactive(input$exec_mat_cor, {
    if (is.null(dados$data) || input$exec_mat_cor == 0 ){ return(NULL)
    }else{
      dados$data %>% select(input$var_mat_cor)
    }
  }, ignoreNULL = FALSE)
  
  output$mat_cor <- renderPlot({
    if (is.null(dataset_mat_cor()) || is.null(dados$data) ){ return(NULL)
    }else{
      if(isolate (input$tipo_matriz) == "circ"){
        corrplot(dataset_mat_cor() %>% drop_na %>% select_if(is.numeric) %>% cor, method = "circle")        
      }
      if(isolate (input$tipo_matriz) == "agrup"){
        corrplot(dataset_mat_cor() %>% drop_na %>% select_if(is.numeric) %>% cor, order= "hclust", addrect = 3, tl.pos="d")  
      }
      if(isolate (input$tipo_matriz) == "med"){
        ggpairs(dataset_mat_cor(), lower = list(continuous = "smooth")) 
      }
    }
  }) 
  
  # Regressão ----
  output$reg_target <- renderUI({
    if (is.null(dados$data)) return(NULL)
    itens=names(dados$data)
    names(itens)= itens
    selectInput("reg_target", "Selecione a variável Target", itens ) 
  })
  
  output$reg_input <- renderUI({
    if (is.null(dados$data)) return(NULL)
    
    itens=names(dados$data)
    names(itens)= itens
    
    if (input$selectall_2%%2 == 0)
    {
      checkboxGroupInput("reg_input", "Selecione as variáveis Input 
                       (a variável Target será desconsideraaa mesmo que seja marcada)", itens, selected = itens ) 
    }
    else
    {
      checkboxGroupInput("reg_input", "Selecione as variáveis Input 
                       (a variável Target será desconsiderama mesmo que seja marcada)", itens) 
    }
  })
  
  # Reativo para atualizar as variaveis (utilizando também isolate() )
  obs_pred <- eventReactive(input$exec_reg, {
    if (is.null(dados$data) ){ return(NULL)
    }else{
      saida <- try(moedor_reg(dados = dados$data,
                              target =  input$reg_target,
                              input_reg = input$reg_input,
                              p = input$reg_p,
                              modelos = input$reg_modelos
      ), silent=TRUE) 
      return( saida )
    }
  }, ignoreNULL = FALSE)
  
  
  output$reg_plot <- renderPlot({
    if (is.null(obs_pred()[[1]]) || input$exec_reg == 0){ return(NULL)
    }else{
      for( i in 2:ncol(obs_pred()[[1]]) ){
        dt_aux <- data.frame(x = obs_pred()[[1]][,1], y = obs_pred()[[1]][,i])
        eval(
          parse(
            text = paste('p',i,' <- ggplot(dt_aux) +
                         geom_point(aes(x=x,y=y)) +
                         geom_line(aes(x=x,y=x), color="blue") +
                         labs(title = names(obs_pred()[[1]])[i], x = "Observado", y = "Predito")', sep='')
          )
        )
      }   
      a <- rep("p",ncol(obs_pred()[[1]])-1)
      b <- 2:ncol(obs_pred()[[1]]) %>% as.character()
      str_aux <- a %+% b
      graf <- str_aux[1]
      if((ncol(obs_pred()[[1]])-1 ) > 1){
      for (j in 2:length(str_aux) ) {
        graf <- graf %+% "," %+% str_aux[j]  
      }
      eval(
        parse(
          text = paste('g <- grid.arrange(',graf,', ncol = 2)', sep='')
        )
      )
      }else{ g <- p2 }
      return(g)
    }
  }) 
  
  output$reg_med <- renderTable({
    
    if (is.null(obs_pred()[[1]]) || input$exec_reg == 0 ){ return(NULL)
    }else{
      aux <- vector()
      for( i in 2:ncol(obs_pred()[[1]]) ){
        #        names(obs_pred()[[1]][,i]) = abs(obs_pred()[[1]][,1]-obs_pred()[[1]][,i]) %>% mean
        v = (obs_pred()[[1]][,1]-obs_pred()[[1]][,i])^2 %>% mean %>% sqrt
        aux <- cbind(aux,v)
      }
      colnames(aux) = names(obs_pred()[[1]])[2:ncol(obs_pred()[[1]])]
      Modelos = colnames(aux)
      REQM = aux[1,]
      Tempo = obs_pred()[[2]][,2]
      aux2 = data.frame(Modelos,REQM,Tempo)
      return(aux2)
    }
  }) 

   output$model_performance <- renderPlot({
     if (is.null(obs_pred()[[1]]) || input$exec_reg == 0){ return(NULL)
     }else{
       g <- plot(obs_pred()[[4]])
       return(g)
     }
   }) 
  
    
  output$var_imp <- renderPlot({
    if (is.null(obs_pred()[[1]]) || input$exec_reg == 0){ return(NULL)
    }else{
      g <- plot(obs_pred()[[3]])
      return(g)
    }
  }) 
  
  
  output$var_imp_out <- renderPrint({
    if (is.null(obs_pred()[[1]]) || input$exec_reg == 0){ return(NULL)
    }else{
      return(obs_pred()[[3]])
    }
  })
  
}

#### Executa o aplicativo ----
shinyApp(ui = ui, server = server)

