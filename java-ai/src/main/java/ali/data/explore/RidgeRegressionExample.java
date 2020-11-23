package ali.data.explore;


import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.*;


import java.util.ArrayList;

import java.util.List;



/**
 * L2岭回归示例
 * https://stackoverflow.com/questions/37639709/how-to-use-l1-regularization-for-logisticregressionwithlbfgs-in-spark-mllib
 *
 */

/**
 * z统计量
 * https://support.minitab.com/zh-cn/minitab/18/help-and-how-to/statistics/basic-statistics/supporting-topics/tests-of-means/what-is-a-z-value/
 */
public class RidgeRegressionExample
{
    public static void main(String[] args)
    {
        SparkSession spark = SparkSession.builder().
                master("spark://titanic:7077").
                appName("Correlation").
                getOrCreate();

        spark.sparkContext().addJar("target/java-ai-1.0-SNAPSHOT.jar");

        Dataset<Row> trainDS = spark.read()
             .option("header", "true")
             .option("delimiter","\t")
             .format("csv")
             .load("src/main/resources/zhengqi_test.txt");

        final String[] columns = trainDS.columns();
        StructField[] lrTrainFieidArray = new StructField[columns.length];

        for (int i = 0; i < columns.length; i++)
        {
            StructField structField = new StructField(columns[i], DataTypes.DoubleType, true, Metadata.empty());
            lrTrainFieidArray[i] = structField;
        }

        StructType lrTrainDsSchema = new StructType(lrTrainFieidArray);

        Dataset<Row> lrTrainDS = trainDS.map(new MapFunction<Row, Row>()
        {
            public Row call(Row row) throws Exception
            {
                List<Double> list = new ArrayList<Double>();
                for (int i = 0; i < columns.length; i++)
                {
                   double data =  Double.parseDouble(row.getString(i));
                   list.add(data);
                }
                Row rowx = RowFactory.create(list.toArray());
                return rowx;
            }
        }, RowEncoder.apply(lrTrainDsSchema));

        trainDS.printSchema();
        System.out.println("------ 转换数据类型，string -> double -----");
        lrTrainDS.printSchema();


//        LinearRegression lr = new LinearRegression();
//        /**
//         * 设置ElasticNet混合参数。当= 0时，惩罚是L2。等于= 1，它是一个L1。等于 0 < alpha < 1，惩罚是L1和L2的组合。默认值0.0是L2惩罚。
//         */
//        lr.setElasticNetParam(0.0);

        spark.stop();


    }
}
