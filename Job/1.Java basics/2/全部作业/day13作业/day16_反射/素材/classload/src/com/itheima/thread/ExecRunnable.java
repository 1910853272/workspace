package com.itheima.thread;

import com.itheima.classloader.HotClassLoader;

import java.io.File;

public class ExecRunnable implements Runnable {

    // 定义要热加载的源文件的包,并通过构造方法进行初始化
    private String hotLoaderSourceFilePackage ;
    public ExecRunnable(String hotLoaderSourceFilePackage) {
        this.hotLoaderSourceFilePackage = hotLoaderSourceFilePackage ;
    }

    @Override
    public void run() {

        while(true) {

            // 定义循环，每隔一秒执行一次
            while(true) {

                // 获取热加载下的源文件的目录中所有的文件
                String sourceFileDirectory = "src/" + hotLoaderSourceFilePackage.replace("." , "/") + "/" ;
                File directory = new File(sourceFileDirectory);
                File[] files = directory.listFiles();
                for(File file : files) {
                    if(file.getName().endsWith(".class")) {

                        // 补全代码


                    }

                }

                try {
                    Thread.sleep(1000);         // 线程休眠1秒
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }

        }

    }

}
