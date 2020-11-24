package ali.data.explore;


import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;

import org.apache.spark.sql.types.*;

import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.components.Marker;
import tech.tablesaw.plotly.traces.ScatterTrace;


import java.util.ArrayList;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.*;


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
             .load("src/main/resources/zhengqi_train.txt");

        final String[] columns = trainDS.columns();
        StructField[] lrTrainFieidArray = new StructField[columns.length];

        for (int i = 0; i < columns.length; i++)
        {
            StructField structField = new StructField(columns[i], DataTypes.DoubleType, true, Metadata.empty());
            lrTrainFieidArray[i] = structField;
        }

        StructType lrTrainDsSchema = new StructType(lrTrainFieidArray);

        System.out.println("------ 转换数据类型，string -> double -----");
        Dataset<Row> lrTrainDS = trainDS.map(new MapFunction<Row, Row>()
        {
            @Override
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

        String[] featuresArray = Arrays.copyOfRange(columns, 0, columns.length - 1);
        VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(featuresArray).setOutputCol("features");
        Dataset<Row> trainvectorDS = assembler.transform(lrTrainDS).select(col("features"),col("V37").as("label"));

        /**
         * 设置ElasticNet混合参数。当= 0时，惩罚是L2。等于= 1，它是一个L1。等于 0 < alpha < 1，惩罚是L1和L2的组合。默认值0.0是L2惩罚。
         */
        LinearRegression lr = new LinearRegression().setMaxIter(10).setElasticNetParam(0.0);
        LinearRegressionModel ridgeModel = lr.fit(trainvectorDS);

        System.out.println("Coefficients: " + ridgeModel.coefficients() + " Intercept: " + ridgeModel.intercept());

        LinearRegressionTrainingSummary trainingSummary = ridgeModel.summary();
        System.out.println("迭代次数: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));

        //残差
//        trainingSummary.residuals().show();

        //标准差
        System.out.println("MSE:"+trainingSummary.meanSquaredError());
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("R2(决定系数)=: " + trainingSummary.r2());
        Dataset<Row> rrDs =  ridgeModel.transform(trainvectorDS);

        rrDs.describe("prediction").show();

        StructField[] fieid = new StructField[]
        {
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("prediction", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("residuals", DataTypes.DoubleType, false, Metadata.empty()),
        };
        StructType schema = new StructType(fieid);

        //残差 y - y_pred
        Dataset<Row> residDS = rrDs.map(new MapFunction<Row, Row>()
        {
            @Override
            public Row call(Row row) throws Exception
            {
                double label = row.getAs("label");
                double prediction = row.getAs("prediction");
                double residuals = label - prediction;
                return RowFactory.create(label,prediction,residuals);
            }
        }, RowEncoder.apply(schema));



        final double residualsMean = residDS.select(mean("residuals").as("residuals")).collectAsList().get(0).getAs("residuals");
        final double residualsStd = residDS.select(stddev("residuals").as("residuals")).collectAsList().get(0).getAs("residuals");

        StructField[] zFieid = new StructField[]
        {
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("prediction", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("residuals", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("z", DataTypes.DoubleType, false, Metadata.empty()),
        };
        final StructType zSchema = new StructType(zFieid);


        /**
         * 此操作解决 java.lang.UnsupportedOperationException: fieldIndex on a Row without schema is undefined.
         */
        Dataset<Row> residDS2=  spark.createDataFrame(residDS.toJavaRDD(),schema);

        Dataset<Row> zDS = residDS2.map(new MapFunction<Row, Row>()
        {
            @Override
            public Row call(Row row) throws Exception
            {
                double label = row.getAs("label");
                double prediction = row.getAs("prediction");
                double residuals = row.getAs("residuals");
                double z = (residuals - residualsMean) / residualsStd;
                return RowFactory.create(label,prediction,residuals ,z);
            }
        }, RowEncoder.apply(zSchema));

        //z > 3
        double sigma = 3.0;
        Dataset<Row> excepDS = zDS.where(abs(col("z")).gt(sigma));

        List<Row> normalList = zDS.select("label", "prediction").collectAsList();
        List<Row> excepList = excepDS.select("label", "prediction").collectAsList();

        double[] normalx = new double[normalList.size()];
        double[] normaly = new double[normalList.size()];
        for (int i = 0; i < normalList.size(); i++)
        {
            normalx[i] = normalList.get(i).getAs("label");
            normaly[i] = normalList.get(i).getAs("prediction");
        }

        double[] exceplx = new double[excepList.size()];
        double[] exceply = new double[excepList.size()];
        for (int i = 0; i < excepList.size(); i++)
        {
            exceplx[i] = excepList.get(i).getAs("label");
            exceply[i] = excepList.get(i).getAs("prediction");
        }

        spark.stop();
        plot(normalx, normaly, exceplx, exceply);
    }

    public static void plot(double[] ax,double[] ay,double[] bx,double[] by)
    {
        Layout layout = Layout.builder().title("Z异常值").build();
        Marker markera = Marker.builder().color("rgb(17, 157, 255)").build();
        Marker markerb = Marker.builder().color("red").build();
        ScatterTrace tracea = ScatterTrace.builder(ax, ay).marker(markera).build();
        ScatterTrace traceb = ScatterTrace.builder(bx, by).marker(markerb).build();

        Plot.show(new Figure(layout, tracea,traceb));
    }


}
