// Databricks notebook source exported at Wed, 26 Oct 2016 13:45:54 UTC
// MAGIC %md # What's new in Spark 2?
// MAGIC * Rerngvit Yanggratoke @ Combient AB 

// COMMAND ----------

// New accesspoint to API is SparkSession
val ss = spark


// COMMAND ----------

// MAGIC %md # Data loading and preparation

// COMMAND ----------

// WARNING: This notebook will save and load using the following basePath, so make sure the directory is usable.
val tmp_folder_path = "/tmp_folder"
dbutils.fs.rm(basePath, recurse=true)
dbutils.fs.mkdirs(basePath)

// COMMAND ----------

// read the dataset from URL and save it to a file
val dataset_url = "https://s3-eu-west-1.amazonaws.com/com-combient-test/ml-meetup/The_world_population_data.csv"
val raw_data = scala.io.Source.fromURL(dataset_url).mkString.split("\n").filter(_ != "")
val tmp_file_path = tmp_folder_path + "/raw_world_population.csv"
ss.sparkContext.parallelize(raw_data).saveAsTextFile(tmp_file_path)
val raw_df = ss.read.option("header", "true").csv(tmp_file_path)
raw_df.printSchema
raw_df.createOrReplaceTempView("raw_world_population")

// COMMAND ----------

// MAGIC %md * Data cleaning

// COMMAND ----------

// Data cleaning process
import org.apache.spark.sql.functions._
val china_pop = sqlContext.sql(" select `2014` from raw_world_population where `Country Name` = 'China' ").collect()(0)(0)
val raw_df = sqlContext.sql(s""" select `Country Name` As name, `2014`, `2015` from raw_world_population
                             WHERE `2014` <= $china_pop
                             ORDER BY `2014` DESC """)
val toLong        = udf[Long, String]( _.toLong)
val cleaned_s1_df = raw_df.filter( ($"2014".isNotNull) && ($"2015".isNotNull))
val cleaned_s2_df = cleaned_s1_df.withColumn("population_2014", toLong($"2014")).withColumn("population_2015", toLong($"2015"))
val cleaned_s3_df = cleaned_s2_df.filter("""name not in ('Central Europe and the Baltics', 'OECD members', 'High income', 'Low income', 'Arab World', 'Europe & Central Asia', 'Post-demographic dividend', 'IDA only',
                    'Sub-Saharan Africa', 'Sub-Saharan Africa (IDA & IBRD countries)', 'Sub-Saharan Africa (excluding high income)', 'Least developed countries: UN classification', 'Pre-demographic dividend', 'Heavily indebted poor countries (HIPC)', 'Latin America & Caribbean', 'Latin America & the Caribbean (IDA & IBRD countries)', 'IDA blend', 'European Union', 'Latin America & Caribbean (excluding high income)', 'Fragile and conflict affected situations', 'Europe & Central Asia (IDA & IBRD countries)', 'Middle East & North Africa', 'Europe & Central Asia (excluding high income)',
                    'Europe & Central Asia (excluding high income)', 'North America', 'Middle East & North Africa', 'Middle East & North Africa (IDA & IBRD countries)',
                    'Middle East & North Africa (excluding high income)'    )""")
val cleaned_df    = cleaned_s3_df.select("name", "population_2014", "population_2015")
val df = cleaned_df.sort($"population_2014".desc)
df.createOrReplaceTempView("country_pop")
df.show(50,false)



// COMMAND ----------

// MAGIC %md # Dataset API example

// COMMAND ----------

case class CountryData(name: String, population_2014: Long, population_2015: Long)
val countryDS = ss.sql(" select * from country_pop ").as[CountryData]
countryDS.filter(c => {c.population_2015 < c.population_2014}).show(5, false)

// COMMAND ----------

// MAGIC %md # Examples of Subqueries

// COMMAND ----------

// Spark 1.6 and 2.0
ss.sql(""" SELECT * 
           FROM (SELECT * FROM country_pop WHERE population_2014 > 100e6) t1 
           ORDER BY population_2014 DESC LIMIT 5""").show(10, false)

// COMMAND ----------

// Only in Spark 2.0
ss.sql(""" SELECT * 
           FROM country_pop 
           WHERE population_2014 IN (select population_2014 from country_pop WHERE population_2014 > 100e6) 
           ORDER BY NAME ASC LIMIT 5 """).show(10, false)

// COMMAND ----------

// MAGIC %md #ML Pipeline persistence across languages (Python -> Scala)

// COMMAND ----------

// MAGIC %md Note that the content in this section is adapted from databrick notebook described in below
// MAGIC * Ref: https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/32296485791277/579709459681615/2883572398008418/latest.html

// COMMAND ----------

// WARNING: This notebook will save and load using the following basePath, so make sure the directory is usable.
val basePath = "/tmp/mlpipe-persistence-example"
dbutils.fs.rm(basePath, recurse=true)
dbutils.fs.mkdirs(basePath)

// COMMAND ----------

// MAGIC %md * Python: A Data scientist -> training pipeline model for image classification of MNIST

// COMMAND ----------

// MAGIC %py
// MAGIC training_df = sqlContext.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
// MAGIC training_df.cache()
// MAGIC training_df.printSchema()
// MAGIC training_df.show()
// MAGIC print "We have %d training images." % training_df.count()

// COMMAND ----------

// MAGIC %py
// MAGIC # Building a pipeline
// MAGIC from pyspark.ml import Pipeline
// MAGIC from pyspark.ml.feature import StandardScaler
// MAGIC from pyspark.ml.classification import RandomForestClassifier
// MAGIC scaler          = StandardScaler(inputCol="features", outputCol="scaledFeatures",
// MAGIC                         withStd=True, withMean=False)
// MAGIC scaler_model    = scaler.fit(training_df)
// MAGIC dataset_scaled  = scaler_model.transform(training_df)
// MAGIC rf_classifier   = RandomForestClassifier(featuresCol="scaledFeatures",labelCol="label",
// MAGIC                                        predictionCol="prediction",probabilityCol="probability",
// MAGIC                                        maxDepth=5,numTrees=100)
// MAGIC ml_pipeline     =  Pipeline(stages=[scaler_model, rf_classifier])
// MAGIC 
// MAGIC # Fitting the pipeline model to the training set
// MAGIC pipeline_model  = ml_pipeline.fit(training_df)
// MAGIC 
// MAGIC # Writing the pipeline to a file
// MAGIC base_path       = "/tmp/mlpipe-persistence"
// MAGIC pipeline_model.save(base_path + "/fitted_pipeline")

// COMMAND ----------

// MAGIC %md Scala: A data engineer -> Take the pipeline model and run it on test data

// COMMAND ----------

// Loading the test dataset
val test_df = sqlContext.read.format("libsvm").option("numFeatures", "784").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")

import org.apache.spark.ml._
val base_path = "/tmp/mlpipe-persistence-example"
// Loading the pipeline
val pipeline = PipelineModel.read.load(base_path + "/fitted_pipeline")

// Testing the pipeline
val predictions = pipeline.transform(test_df)
predictions.show

// COMMAND ----------

// MAGIC %md # Approximate DataFrame  Statistics Functions

// COMMAND ----------

// MAGIC %md * Approximate Quantile

// COMMAND ----------

df.stat.approxQuantile(
       col="population_2014", 
       probabilities=Array(0.25, 0.5, 0.75, 0.99), 
       relativeError=0.05)

// COMMAND ----------

// MAGIC %md * Bloom filter

// COMMAND ----------

val bfCountryName = df.stat.bloomFilter(colName="name", expectedNumItems=300, fpp =0.01)
println(bfCountryName.mightContain("Sweden"))
println(bfCountryName.mightContain("Unknow Country"))

// COMMAND ----------

// MAGIC %md * CountminSketch

// COMMAND ----------

val nameSketch = df.stat.countMinSketch(colName="name", eps = 0.001, 
                             confidence = 0.99, seed = 42) 
println(nameSketch.estimateCount("Sweden"))
println(nameSketch.estimateCount("Unknown Country"))


// COMMAND ----------


