package io.skymind.issue;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * Created by kepricon on 17. 4. 20.
 */
public class Issue3243 {
    public static void main(String[] args) throws Exception {

        Model model = new MultiLayerNetwork(new LeNet(224, 224, 3, 200, 1234, 1).conf());
        model.init();

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"),  "dl4j_tiny_16batch_224x224/dl4j_tinyimagenet_train/");
        File savedDataPath = new File(path);
        if (savedDataPath.exists() == false){
            new TinyImageNetDataSetBuilder().run(null);
        }

        DataSetIterator iter = new ExistingMiniBatchDataSetIterator(savedDataPath);

        ParallelWrapper pw = new ParallelWrapper.Builder<>(model)
                .prefetchBuffer(4)
                .reportScoreAfterAveraging(true)
                .averagingFrequency(10)
                .useLegacyAveraging(false)
                .useMQ(true)
                .workers(4)
                .averageUpdaters(false)
                .build();

        pw.fit(iter);
    }
}
