package ali.data.explore;

import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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

        Dataset<Row> trainDS = spark.read()
             .option("header", "true")
             .option("delimiter","\t")
             .format("csv")
             .load("src/main/resources/zhengqi_test.txt");

        trainDS.show();

        LinearRegression lr = new LinearRegression();
        /**
         * 设置ElasticNet混合参数。当= 0时，惩罚是L2。等于= 1，它是一个L1。等于 0 < alpha < 1，惩罚是L1和L2的组合。默认值0.0是L2惩罚。
         */
        lr.setElasticNetParam(0.0);

        spark.stop();


    }
}
