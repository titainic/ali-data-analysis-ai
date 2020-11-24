package ali.data.explore;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LRDemo
{
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("spark://titanic:7077").appName("LRDemo").getOrCreate();
        // $example on$
        Dataset<Row> data = spark.read().format("libsvm").load("src/main/resources/sample_linear_regression_data.txt");
        data.show();
        // Prepare training and test data.
        Dataset<Row>[] splits = data.randomSplit(new double[] { 0.9, 0.1 }, 12345);
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];
        LinearRegression lr = new LinearRegression();
        // the evaluator.
        ParamMap[] paramGrid = new ParamGridBuilder().addGrid(lr.regParam(), new double[] { 0.1, 0.01 }).addGrid(lr.fitIntercept()).addGrid(lr.elasticNetParam(), new double[] { 0.0, 0.5, 1.0 }).build();
        // A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
        TrainValidationSplit trainValidationSplit = new TrainValidationSplit().setEstimator(lr).setEvaluator(new RegressionEvaluator()).setEstimatorParamMaps(paramGrid).setTrainRatio(0.8);
        // Run train validation split, and choose the best set of parameters.
        TrainValidationSplitModel model = trainValidationSplit.fit(training);
        // that performed best.
        model.transform(test).select("features", "label", "prediction").show();
        // $example off$
        spark.stop();
    }
}
