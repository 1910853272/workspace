package com.itheima.d4_synchronized_code;

public class DrawThread extends Thread{
    private Account acc;
    public DrawThread(Account acc, String name){
        super(name);
        this.acc = acc;
    }
    @Override
    public void run() {
        // 取钱(小明，小红)
        acc.drawMoney(100000);
    }
}
