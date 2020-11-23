package ali.data.explore;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.stat.Correlation;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.col;

/**
 * 数据相关性
 *
 *
 *
 * 针对【连续、正态分布、线性】数据，采用pearson相关系数；
 * 针对【非线性的、非正态】数据，采用spearman相关系数；
 * 针对【分类变量、无序】数据，采用Kendall
 * 相关系数。一般来讲，线性数据采用pearson，否则选择spearman，如果是分类的则用kendall。
 * https://www.jianshu.com/p/f9304da68d98
 * https://blog.csdn.net/u013230189/article/details/82316574
 */
public class CorrelationSpark
{
    public static void main(String[] args)
    {
        init();
    }

    public static void init()
    {
        SparkSession spark = SparkSession.builder().
            master("spark://titanic:7077").
            appName("Correlation").
            getOrCreate();

        Row x = RowFactory.create(65, 72, 78, 65, 72, 70, 65, 68);
        Row y = RowFactory.create(72, 69, 79, 69, 84, 75, 60, 73);
        List<Row> data = Arrays.asList(x, y);

        StructType schema = new StructType(new StructField[] {
            new StructField("v1", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v2", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v3", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v4", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v5", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v6", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v7", DataTypes.IntegerType, false, Metadata.empty()),
            new StructField("v8", DataTypes.IntegerType, false, Metadata.empty())});

        Dataset<Row> dataDS = spark.createDataFrame(data, schema);

        //数据行，转为向量
        String[] col = {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"};
        VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(col).setOutputCol("features");
        Dataset<Row> vectorDS = assembler.transform(dataDS).select(col("features"));

        vectorDS.show();

        Row corrPearson = Correlation.corr(vectorDS, "features").head();
        Matrix cMatrix = (Matrix) corrPearson.get(0);
        System.out.println("Pearson correlation matrix:\n" +cMatrix.toString(8,500));

        Row corrSpearman = Correlation.corr(vectorDS, "features","spearman").head();
        Matrix sMatrix = (Matrix) corrSpearman.get(0);
        System.out.println("Spearman correlation matrix:\n" +sMatrix.toString(8,500));

        spark.stop();
    }
}
