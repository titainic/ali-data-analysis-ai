package ali.data.explore;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.stat.Correlation;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.util.Arrays;
import java.util.List;

import static org.apache.spark.sql.functions.col;


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

        StructType schema = new StructType(new StructField [] {
                new StructField("v1", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v2", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v3", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v4", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v5", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v6", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v7", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("v8", DataTypes.IntegerType, false, Metadata.empty())
        });

        Dataset<Row> dataDS = spark.createDataFrame(data, schema);

        //数据行，转为向量
        String[] col = {"v1", "v2","v3", "v4","v5", "v6","v7", "v8"};
        VectorAssembler assembler = new VectorAssembler();
        assembler.setInputCols(col).setOutputCol("features");
        Dataset<Row> tmpds = assembler.transform(dataDS).select(col("features"));

        Row corrPearson = Correlation.corr(tmpds, "features").head();
        System.out.println("Pearson correlation matrix:\n" + corrPearson.get(0).toString());

        Row corrSpearman = Correlation.corr(tmpds, "features","spearman").head();
        System.out.println("Spearman correlation matrix:\n" +corrSpearman.get(0).toString());

        spark.stop();

    }
}
