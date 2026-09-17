<?php
// 僅限本機沙盒：這支檔案會讓 Adminer 自動登入，帳密寫死在 credentials()。
// 正式環境或任何對外的服務都不可以使用。

function adminer_object() {
    class AutoLoginAdminer extends Adminer {
        function login($login, $password) {
            return true;
        }
        function credentials() {
            return array('heartlink-pg', 'heartlink', 'heartlink');
        }
        function loginForm() {
            parent::loginForm();
            echo "<script>document.querySelector('form').submit();</script>";
        }
    }
    return new AutoLoginAdminer;
}

if (!isset($_GET['pgsql'])) {
    $_GET['pgsql'] = 'heartlink-pg';
}

require('adminer.php');
